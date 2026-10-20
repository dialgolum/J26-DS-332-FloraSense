"""C1 plant identification and advisory service API entry point."""

import io
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from PIL import Image
from services.c1_plant_id_advisory.vision.inference.classifier import PlantIdentifier

app = FastAPI(title="FloraSense C1")
identifier = PlantIdentifier()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return identifier.identify(img)

app.mount("/", StaticFiles(directory="frontend", html=True), name="ui")
