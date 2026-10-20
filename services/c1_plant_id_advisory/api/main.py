import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from services.c1_plant_id_advisory.vision.inference.classifier import PlantIdentifier

app = FastAPI(title="FloraSense C1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
identifier = PlantIdentifier()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return identifier.identify(img)