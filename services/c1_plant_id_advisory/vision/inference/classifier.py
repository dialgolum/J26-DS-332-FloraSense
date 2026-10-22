"""Plant variety classifier using vision embeddings and prototypes."""
import torch, open_clip, joblib
from PIL import Image

class PlantIdentifier:
    def __init__(self, ckpt="models/checkpoints/classifier.joblib", threshold=0.5):
        self.model, _, self.pre = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        self.model.eval()
        self.clf = joblib.load(ckpt)
        self.threshold = threshold

    @torch.no_grad()
    def identify(self, img: Image.Image):
        x = self.pre(img.convert("RGB")).unsqueeze(0)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        p = self.clf.predict_proba(f.numpy())[0]
        order = p.argsort()[::-1][:3]
        top = [(str(self.clf.classes_[i]), float(p[i])) for i in order]
        return {"label": top[0][0], "confidence": top[0][1],
                "is_unknown": top[0][1] < self.threshold, "top3": top}