import logging
from pathlib import Path
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
connected_clients = set()


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected, total={len(connected_clients)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected, total={len(connected_clients)}")


@app.post("/ingest")
async def ingest(request: Request):
    payload = await request.json()
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.add(client)
    connected_clients.difference_update(disconnected)
    return {"delivered": len(connected_clients)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
