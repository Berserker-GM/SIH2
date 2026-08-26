"""
NORMATIVE // DEV 4 — WebSocket Connection Manager
Tracks active connections and broadcasts messages.
"""
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def send(self, ws: WebSocket, data: str):
        """Send to a single connection; remove if it's dead."""
        try:
            await ws.send_text(data)
        except Exception:
            self.disconnect(ws)

    async def broadcast(self, data: str):
        """Send to every connected client."""
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
