import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baseline"))

import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from model import StorifaiBaseline

app = FastAPI(title="Storifai Baseline API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CHECKPOINT = "baseline/models/checkpoint.pth"
DEVICE = "cpu"
MAX_LEN = 25

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = None
vocab = None

def load_model():
    global model, vocab
    if not os.path.exists(CHECKPOINT):
        print("No checkpoint found - demo mode")
        return
    print("Loading model...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    vocab = ckpt["vocab"]
    model = StorifaiBaseline(len(vocab), 256, 512, 1).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded. Vocab: {len(vocab)} words")

def generate_sentence(img_tensor):
    if model is None or vocab is None:
        return None
    idx2word = {v: k for k, v in vocab.word2idx.items()}
    with torch.no_grad():
        img = img_tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)
        dummy = torch.zeros(1, 1, MAX_LEN, dtype=torch.long).to(DEVICE)
        out = model(img, dummy)
        ids = out[0, 0].argmax(dim=-1).tolist()
        words = []
        for wid in ids:
            w = idx2word.get(wid, "")
            if w in ("<END>", "<PAD>", ""):
                break
            if w != "<START>":
                words.append(w)
    return " ".join(words).capitalize() + "." if words else None

DEMOS = [
    ["We arrived just as the light was perfect.", "Everyone was full of energy and smiles.", "We explored every corner together.", "By afternoon we found our favorite spot.", "It was a day we will never forget."],
    ["The morning started quietly.", "We wandered without a plan.", "Strangers became friends.", "The afternoon light made everything golden.", "We left feeling grateful."]
]

@app.get("/")
def root():
    return {"status": "ok", "model": "loaded" if model else "demo"}

@app.post("/generate")
async def generate(images: List[UploadFile] = File(...)):
    if len(images) < 3 or len(images) > 5:
        raise HTTPException(400, "Upload 3-5 photos")
    story = []
    for img_file in images:
        data = await img_file.read()
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            tensor = transform(img)
            sentence = generate_sentence(tensor)
            if sentence:
                story.append(sentence)
        except:
            pass
    if not story:
        story = random.choice(DEMOS)[:len(images)]
    return {"story": story, "model": "baseline" if model else "demo"}

@app.on_event("startup")
async def startup():
    load_model()