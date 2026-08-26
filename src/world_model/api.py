"""
World-model HTTP API for the SIH dashboard.

No Redis. Frozen LSTM only.

    python -m src.world_model.api
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.world_model.explain import LSTM_ATTACK_THRESHOLD
from src.world_model.service import (
    WorldModelRuntime,
    load_frozen_runtime,
    resolve_local_csv,
    resolve_local_pcap,
    list_local_csv,
    local_cic_dir,
    LOCAL_CIC_DIR_REL,
    MAX_UPLOAD_BYTES,
)

_DEFAULT_RUNTIME: WorldModelRuntime | None = None


class ForecastBody(BaseModel):
    history: list[list[float]] = Field(..., description="Scaled (8, 32) history ending at S_t")


def get_runtime() -> WorldModelRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = load_frozen_runtime()
    return _DEFAULT_RUNTIME


def create_app(runtime: WorldModelRuntime | None = None) -> FastAPI:
    app = FastAPI(title="Normative World Model", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    box = {"rt": runtime}

    def rt() -> WorldModelRuntime:
        return box["rt"] if box["rt"] is not None else get_runtime()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "service": "world-model"}

    @app.get("/api/world-model/status")
    def status() -> dict:
        return rt().status()

    @app.get("/api/world-model/local-csv")
    def local_csv_list() -> dict[str, Any]:
        """CSVs on disk under data/raw (drop files into data/raw/CIC-IDS2018)."""
        local_cic_dir()
        return {"dir": LOCAL_CIC_DIR_REL, "files": list_local_csv()}

    @app.get("/api/world-model/examples")
    def examples() -> dict:
        items = [rt().example(i) for i in range(len(rt().examples))]
        return {
            "n": len(items),
            "attack_threshold": LSTM_ATTACK_THRESHOLD,
            "day": "2018-03-01",
            "examples": items,
        }

    @app.get("/api/world-model/forecast")
    def forecast_example(i: int = Query(0, ge=0)) -> dict:
        try:
            return rt().example(i)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/world-model/forecast")
    def forecast_live(body: ForecastBody) -> dict[str, Any]:
        hist = body.history
        try:
            return rt().forecast(hist)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/world-model/upload")
    def upload_csv(
        file: UploadFile = File(...),
        pcap: UploadFile | None = File(None),
        slice_mode: str = Query("latest"),
    ) -> dict[str, Any]:
        name = file.filename or "upload.csv"
        if not name.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload a CIC-IDS2018 .csv file")
        tmp_path = None
        pcap_tmp = None
        try:
            written = 0
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp_path = Path(tmp.name)
                while True:
                    buf = file.file.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    written += len(buf)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "CSV is larger than 80 MB. Tuesday-20-02-2018 (LOIC-HTTP) is ~4 GB "
                                "and not time-sorted, so a browser upload will not finish. "
                                "Keep the file in data/raw/CIC-IDS2018 and pick it from the local CSV dropdown.",
                            ),
                        )
                    tmp.write(buf)
            if pcap is not None and pcap.filename:
                pname = pcap.filename.lower()
                if not pname.endswith((".pcap", ".pcapng", ".cap")):
                    raise HTTPException(status_code=400, detail="Optional packet file must be .pcap / .pcapng")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                    pcap_tmp = Path(tmp.name)
                    pcap_written = 0
                    while True:
                        buf = pcap.file.read(8 * 1024 * 1024)
                        if not buf:
                            break
                        pcap_written += len(buf)
                        if pcap_written > MAX_UPLOAD_BYTES:
                            raise HTTPException(status_code=413, detail="PCAP is larger than 80 MB")
                        tmp.write(buf)
            return rt().infer_from_csv(
                tmp_path, source_file=name, pcap_path=pcap_tmp, slice_mode=slice_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            if pcap_tmp is not None:
                pcap_tmp.unlink(missing_ok=True)

    @app.post("/api/world-model/local")
    def local_csv(
        name: str = Query("ddos-loic-http"),
        pcap: str | None = Query(None),
        slice_mode: str = Query("latest"),
    ) -> dict[str, Any]:
        """Run the frozen LSTM on a CIC CSV already under data/raw (no 4 GB upload)."""
        try:
            path = resolve_local_csv(name)
            pcap_path = resolve_local_pcap(pcap) if pcap else None
            out = rt().infer_from_csv(
                path, source_file=path.name, pcap_path=pcap_path, slice_mode=slice_mode,
            )
            out["source"] = "local"
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/world-model/sample")
    def sample_csv() -> dict[str, Any]:
        """Same live path as upload, using synthetic CIC-shaped rows (not saved 01 Mar JSON)."""
        from src.world_model.prepare_dataset import generate_demo_flows

        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.close()
            tmp_path = Path(tmp.name)
            generate_demo_flows(n_windows=24, flows_per_window=3).to_csv(tmp_path, index=False)
            out = rt().infer_from_csv(tmp_path, source_file="synthetic_demo.csv", slice_mode="latest")
            out["source"] = "sample"
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("src.world_model.api:app", host="0.0.0.0", port=8001, reload=False)


if __name__ == "__main__":
    main()
