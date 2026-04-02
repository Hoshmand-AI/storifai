import os, sys, random
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(title="Storifai Baseline API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEMOS = [
    ["We arrived just as the light was perfect.", "Everyone was full of energy and smiles.", "We explored every corner together.", "By afternoon we found our favorite spot.", "It was a day we will never forget."],
    ["The morning started quietly.", "We wandered without a plan.", "Strangers became friends.", "The afternoon light made everything golden.", "We left feeling grateful."]
]

@app.get("/")
def root():
    return {"status": "ok", "model": "demo"}

@app.post("/generate")
async def generate(images: List[UploadFile] = File(...)):
    if len(images) < 3 or len(images) > 5:
        raise HTTPException(400, "Upload 3-5 photos")
    demo = random.choice(DEMOS)
    return {"story": demo[:len(images)], "model": "demo"}
