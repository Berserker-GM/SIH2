#!/usr/bin/env python3
"""World-model FastAPI wiring: frozen LSTM, no Redis required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Payload serialization ----------------------------------------------")
from src.world_model.service import WorldModelRuntime, jsonable
from src.world_model.dataset import Scaler
from src.world_model.explain import LSTM_ATTACK_THRESHOLD
from src.world_model.labels import STAGE_ID
from src.world_model.mitre_decode import fit_stage_decoder
from src.world_model.model import LSTMWorldModel

blob = jsonable({"a": np.float32(1.5), "b": np.zeros((2, 3), dtype=np.float32)})
check("jsonable scalar", blob["a"] == 1.5)
check("jsonable nested list", blob["b"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


print("\n-- Runtime.forecast with injected frozen LSTM -------------------------")


class CopyLast(nn.Module):
    def forward(self, x):
        last = x[:, -1, :]
        return last, x.new_zeros(x.size(0))


X = np.zeros((40, 32), dtype=np.float32)
y = np.zeros(40, dtype=np.int32)
X[20:, 2] = 3.0
y[20:] = STAGE_ID["lateral_movement"]
decoder = fit_stage_decoder(X, y)
scaler = Scaler(mean=np.zeros(32), std=np.ones(32))
rt = WorldModelRuntime(
    model=CopyLast(),
    scaler=scaler,
    decoder=decoder,
    examples=[{"narrative": "demo", "stage": "lateral_movement"}],
)
hist = np.zeros((8, 32), dtype=np.float32)
hist[:, 2] = 3.0
out = rt.forecast(hist)
check("forecast something_bad is bool", isinstance(out["something_bad"], bool))
check("threshold stays 0.15", out["attack_threshold"] == LSTM_ATTACK_THRESHOLD)
check("has narrative", "narrative" in out and len(out["narrative"]) > 10)
check("example cycling", rt.example(0)["stage"] == "lateral_movement")
check("example wrap", rt.example(9)["stage"] == "lateral_movement")


print("\n-- FastAPI routes (no Redis) ------------------------------------------")
from fastapi.testclient import TestClient
from src.world_model.api import create_app

app = create_app(runtime=rt)
client = TestClient(app)
r = client.get("/api/world-model/status")
check("status 200", r.status_code == 200)
check("status reports frozen", r.json().get("frozen") is True)
check("status input_dim 32", r.json().get("input_dim") == 32)
spec = r.json().get("packet_level_spec") or []
check("status has packet_level_spec", len(spec) == 10)
check("status spec starts at index 22", spec and spec[0].get("index") == 22 and spec[0].get("name") == "pkt_len_mean")
check("status spec ends at retransmit_count", spec and spec[-1].get("index") == 31 and spec[-1].get("name") == "retransmit_count")
check("status local_csv is list", isinstance(r.json().get("local_csv"), list))
check("status local_csv_dir", r.json().get("local_csv_dir") == "data/raw/CIC-IDS2018")

listed = client.get("/api/world-model/local-csv")
check("local-csv 200", listed.status_code == 200)
listed_body = listed.json() if listed.status_code == 200 else {}
check("local-csv dir", listed_body.get("dir") == "data/raw/CIC-IDS2018")
check("local-csv files is list", isinstance(listed_body.get("files"), list))
if listed_body.get("files"):
    row = listed_body["files"][0]
    check("local-csv row has rel/name/bytes", "rel" in row and "name" in row and "bytes" in row)
    check("local-csv rows are csv", str(row.get("name") or "").lower().endswith(".csv"))

ex = client.get("/api/world-model/examples")
check("examples 200", ex.status_code == 200)
check("examples list", len(ex.json()["examples"]) >= 1)

one = client.get("/api/world-model/forecast?i=0")
check("forecast GET 200", one.status_code == 200)
check("GET has stage", "stage" in one.json())

bad = client.post("/api/world-model/forecast", json={"history": [[0.0] * 32]})
check("POST bad shape 400", bad.status_code == 400)

ok = client.post("/api/world-model/forecast", json={"history": np.zeros((8, 32)).tolist()})
check("POST 8x32 200", ok.status_code == 200)
check("POST returns why lists", isinstance(ok.json().get("why_attack"), list))

src = (PROJECT_ROOT / "src" / "world_model" / "model.py").read_text(encoding="utf-8")
check("model.py still frozen architecture", "num_layers=2" in src or "num_layers: int = 2" in src)


print("\n-- CSV upload → windows → forecast ------------------------------------")
from src.world_model.prepare_dataset import generate_demo_flows

demo_csv = generate_demo_flows(n_windows=24, flows_per_window=3).to_csv(index=False).encode("utf-8")
up = client.post(
    "/api/world-model/upload",
    files={"file": ("demo.csv", demo_csv, "text/csv")},
)
check("upload 200", up.status_code == 200, str(up.status_code) + " " + up.text[:180])
body = up.json() if up.status_code == 200 else {}
check("upload source is upload", body.get("source") == "upload")
check("upload has attack_probability", "attack_probability" in body)
check("upload feature_names 32", len(body.get("feature_names") or []) == 32)
hist_states = body.get("history_states") or []
check(
    "upload history 8x32",
    isinstance(hist_states, list) and len(hist_states) == 8 and len(hist_states[0]) == 32,
)
check("upload has s_t dict", isinstance(body.get("s_t"), dict) and "bytes_fwd_sum" in (body.get("s_t") or {}))
check("upload has timeline of 6 steps", isinstance(body.get("timeline"), list) and len(body.get("timeline") or []) == 6)
if body.get("timeline"):
    step = body["timeline"][0]
    check("timeline has seconds_ahead", step.get("seconds_ahead") == 5)
    check("timeline has timestamp", "timestamp" in step and "timestamp_iso" in step)
    check("timeline has step attack P", "attack_probability" in step)
    check("timeline has step stage", "stage" in step and "technique_id" in step)
check("upload MITRE stage present", body.get("stage") in {
    "benign", "reconnaissance", "initial_access", "lateral_movement",
    "command_and_control", "exfiltration", "impact",
})
check("upload why_attack", isinstance(body.get("why_attack"), list) and len(body["why_attack"]) >= 1)
check("upload why_stage", isinstance(body.get("why_stage"), list) and len(body["why_stage"]) >= 1)
check("upload why_change", isinstance(body.get("why_change"), list) and len(body["why_change"]) >= 1)
check("upload forecast_timestamps length 6", len(body.get("forecast_timestamps") or []) == 6)
sg = body.get("state_graph") or {}
check("upload has state_graph", isinstance(body.get("state_graph"), dict) and "mode" in sg)
check("upload graph is service mode", sg.get("mode") == "service")
check("upload graph has nodes+edges", len(sg.get("nodes") or []) >= 2 and len(sg.get("edges") or []) >= 1)
check("upload graph leaves 32-d intact", len(body.get("feature_names") or []) == 32)
up_spec = body.get("packet_level_spec") or []
check("upload packet_level_spec length 10", len(up_spec) == 10)
check(
    "upload spec order 22–31",
    [row.get("name") for row in up_spec] == [
        "pkt_len_mean", "pkt_len_std", "pkt_len_max",
        "init_fwd_win_mean", "init_bwd_win_mean", "fwd_psh_flags_sum",
        "port_scan_score", "ttl_variance", "ip_fragment_flags", "retransmit_count",
    ],
)
check("upload without pcap has no pcap_stats", "pcap_stats" not in body)
s_t = body.get("s_t") or {}
check("upload S_t PCAP stubs stay 0", s_t.get("ttl_variance") == 0.0 and s_t.get("retransmit_count") == 0.0)
check("upload slice_mode latest", body.get("slice_mode") == "latest")
check("upload s_t_attack_now is 0 or 1", body.get("s_t_attack_now") in {0, 1})
check("upload CIC bin at S_t is benign", body.get("s_t_attack_now") == 0)


print("\n-- Optional PCAP sidecar on upload ------------------------------------")
from datetime import datetime, timezone
from scapy.layers.inet import IP, TCP, fragment
from scapy.layers.l2 import Ether
from scapy.packet import Raw
from scapy.utils import wrpcap

t_last = datetime(2018, 2, 28, 10, 1, 50, tzinfo=timezone.utc).timestamp()
pcap_pkts = []
p1 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2", ttl=64) / TCP(sport=1234, dport=80, seq=100, flags="A") / Raw(b"abc")
p1.time = t_last
pcap_pkts.append(p1)
p2 = Ether() / IP(src="10.0.0.1", dst="10.0.0.2", ttl=128) / TCP(sport=1234, dport=80, seq=100, flags="A") / Raw(b"abc")
p2.time = t_last + 0.1
pcap_pkts.append(p2)
big = Ether() / IP(src="10.0.0.3", dst="10.0.0.4", ttl=64) / TCP(sport=1, dport=2, seq=1, flags="S") / Raw(b"X" * 2000)
frags = fragment(big, fragsize=1000)
for i, frag in enumerate(frags):
    frag.time = t_last + 0.2 + 0.01 * i
    pcap_pkts.append(frag)
pcap_tmp = PROJECT_ROOT / "data" / "processed" / "_upload_sidecar.pcap"
pcap_tmp.parent.mkdir(parents=True, exist_ok=True)
wrpcap(str(pcap_tmp), pcap_pkts)
pcap_bytes = pcap_tmp.read_bytes()
up_pcap = client.post(
    "/api/world-model/upload",
    files={
        "file": ("demo.csv", demo_csv, "text/csv"),
        "pcap": ("demo.pcap", pcap_bytes, "application/octet-stream"),
    },
)
pcap_tmp.unlink(missing_ok=True)
check("upload+pcap 200", up_pcap.status_code == 200, str(up_pcap.status_code) + " " + up_pcap.text[:180])
pcap_body = up_pcap.json() if up_pcap.status_code == 200 else {}
ps = pcap_body.get("pcap_stats") or {}
check("upload+pcap has sidecar", isinstance(pcap_body.get("pcap_stats"), dict))
check("sidecar ttl_variance > 0", float(ps.get("ttl_variance") or 0) > 0.0, str(ps.get("ttl_variance")))
check("sidecar fragments counted", float(ps.get("ip_fragment_flags") or 0) >= 1.0)
check("sidecar retransmit counted", float(ps.get("retransmit_count") or 0) >= 1.0)
check("sidecar not injected into S_t", ps.get("injected_into_st") is False)
st2 = pcap_body.get("s_t") or {}
check(
    "LSTM S_t still has PCAP zeros",
    st2.get("ttl_variance") == 0.0 and st2.get("ip_fragment_flags") == 0.0 and st2.get("retransmit_count") == 0.0,
)

sample = client.post("/api/world-model/sample")
check("sample 200", sample.status_code == 200, str(sample.status_code) + " " + sample.text[:180])
sample_body = sample.json() if sample.status_code == 200 else {}
check("sample uses live path", sample_body.get("source") == "sample")
check("sample has 8x32 history", len(sample_body.get("history_states") or []) == 8)

replay = client.get("/api/world-model/forecast?i=0")
check("replay still works after upload", replay.status_code == 200 and replay.json().get("stage") == "lateral_movement")

no_ts = client.post(
    "/api/world-model/upload",
    files={"file": ("bad.csv", b"Dst Port,Label\n80,Benign\n", "text/csv")},
)
check("missing timestamp is 400", no_ts.status_code == 400)

empty = client.post("/api/world-model/upload")
check("missing file is 422 or 400", empty.status_code in {400, 422})

import src.world_model.api as api_mod
old_max = api_mod.MAX_UPLOAD_BYTES
api_mod.MAX_UPLOAD_BYTES = 64
over = client.post(
    "/api/world-model/upload",
    files={"file": ("huge.csv", b"x" * 200, "text/csv")},
)
api_mod.MAX_UPLOAD_BYTES = old_max
check("oversize upload is 413", over.status_code == 413)

missing_local = client.post("/api/world-model/local?name=does-not-exist.csv")
check("missing local csv is 400", missing_local.status_code == 400)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  World-model API wiring OK.\n")
