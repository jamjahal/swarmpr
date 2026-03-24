"""FastAPI server for SwarmPR.

Provides REST and WebSocket endpoints for pipeline execution,
metrics retrieval, and real-time event streaming. This is the
integration point for future React frontend.
"""

import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from swarmpr.orchestrator.events import Event

app = FastAPI(
    title="SwarmPR API",
    description="Multi-agent pipeline API for task → PR automation",
    version="0.1.0",
)

# In-memory state for active connections and pipeline runs.
_active_connections: list[WebSocket] = []
_pipeline_history: list[dict] = []


class RunRequest(BaseModel):
    """Request body for triggering a pipeline run.

    Attributes:
        task: The task description.
        repo_path: Path to the target repository.
        config_path: Path to the SwarmPR config file.
    """

    task: str
    repo_path: str
    config_path: str = "config.yaml"


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status.
        version: SwarmPR version.
    """

    status: str = "ok"
    version: str = "0.1.0"


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Service health status and version.
    """
    return HealthResponse()


@app.get("/history")
async def get_history() -> list[dict]:
    """Return the history of pipeline runs.

    Returns:
        List of past pipeline run summaries.
    """
    return _pipeline_history


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time event streaming.

    Clients connect here to receive live pipeline events as JSON.
    This is the integration point for the future React dashboard.

    Args:
        websocket: The WebSocket connection.
    """
    await websocket.accept()
    _active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive, listen for client messages.
            data = await websocket.receive_text()
            # Client can send 'ping' to keep alive.
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _active_connections.remove(websocket)


async def broadcast_event(event: Event) -> None:
    """Broadcast a pipeline event to all connected WebSocket clients.

    Args:
        event: The event to broadcast.
    """
    event_data = json.dumps({
        "event_type": event.event_type.value,
        "message": event.message,
        "stage": event.stage.value if event.stage else None,
        "data": event.data,
        "timestamp": event.timestamp.isoformat(),
    })

    disconnected = []
    for ws in _active_connections:
        try:
            await ws.send_text(event_data)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        _active_connections.remove(ws)
