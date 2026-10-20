"""FastAPI application for the FloraSense C3 quality component."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router
from .automatic_routes import router as automatic_router

app = FastAPI(title="FloraSense C3 Quality Assessment API", version="1.0.0")

origins = [
    origin.strip()
    for origin in os.getenv(
        "FLORASENSE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.exception_handler(HTTPException)
async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        content={"data": None, "error": {"code": f"HTTP_{exc.status_code}", "message": message}},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    details = "; ".join(" ".join(str(part) for part in error["loc"]) + ": " + error["msg"]
                         for error in exc.errors())
    return JSONResponse(
        status_code=422,
        content={"data": None, "error": {"code": "VALIDATION_ERROR", "message": details}},
    )


@app.get("/health")
def health() -> dict:
    return {"data": {"status": "ok", "service": "c3-quality-assessment"}, "error": None}


app.include_router(router)
app.include_router(automatic_router)
