"""
FastAPI entrypoint for Component 2.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from services.c2_disease_detection import config
from services.c2_disease_detection.api.routes import diagnose, register_disease
from services.c2_disease_detection.api.schemas.responses import HealthResponse
from services.c2_disease_detection.staging.review_queue import ReviewQueue
from services.c2_disease_detection.vision.inference.classifier import DiseaseEngine

log = logging.getLogger("c2")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.review_queue = ReviewQueue()
    app.state.engine = None
    try:
        app.state.engine = DiseaseEngine.from_disk()
        log.info("Model loaded: %d prototype classes", len(app.state.engine.store))
    except FileNotFoundError:
        log.warning("Checkpoint %s not found. Run training first.", config.CHECKPOINT_PATH)
    yield


app = FastAPI(
    title="FloraSense C2: Ornamental Plant Disease Detection",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(diagnose.router)
app.include_router(diagnose.review_router)
app.include_router(register_disease.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request):
    engine = request.app.state.engine
    return HealthResponse(
        status="ok" if engine else "model_not_loaded",
        model_loaded=engine is not None,
        prototypes_loaded=len(engine.store) if engine else 0,
        tau=config.TAU,
        temperature=engine.temperature if engine else None,
        review_pending=request.app.state.review_queue.stats()["pending"],
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")


if (config.SERVICE_DIR / "ui").exists():
    app.mount("/ui", StaticFiles(directory=config.SERVICE_DIR / "ui", html=True), name="ui")