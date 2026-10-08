"""C1 plant identification and advisory service API entry point."""

from fastapi import FastAPI

app = FastAPI(title="C1 Plant ID & Advisory Service")


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
