import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from services.c1_plant_id_advisory.vision.inference.classifier import PlantIdentifier
from services.c1_plant_id_advisory.chatbot.advisor import Advisor

app = FastAPI(title="FloraSense C1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

identifier = PlantIdentifier()
advisor = Advisor()


class ChatIn(BaseModel):
    question: str
    variety: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return identifier.identify(img)


@app.post("/chat")
def chat(body: ChatIn):
    try:
        text, sources = advisor.answer(body.question, body.variety)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"answer": text, "sources": sources}