import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import requests
from PIL import Image
from io import BytesIO
import json
import os
import time
from collections import defaultdict

from dataloader import VISTDataset, Vocabulary
from model import StorifaiBaseline

def fetch_flickr_image(flickr_id, transform, size=224):
    url = f"https://live.staticflickr.com//{flickr_id}_b.jpg"
    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return transform(img)
    except:
        return torch.zeros(3, size, size)

TRAIN_JSON  = "shared/dataset/sis/train.story-in-sequence.json"
CHECKPOINT  = "baseline/models/checkpoint.pth"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
EMBED_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 1
BATCH_SIZE  = 4
NUM_EPOCHS  = 5
LR          = 3e-4
MAX_STORIES = 500

os.makedirs("baseline/models", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

print("Loading dataset...")
full_dataset = VISTDataset(TRAIN_JSON)
vocab = full_dataset.vocab
subset = Subset(full_dataset, range(min(MAX_STORIES, len(full_dataset))))
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Training on {len(subset)} stories | Device: {DEVICE}")

model = StorifaiBaseline(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])

def train_epoch(epoch):
    model.train()
    total_loss = 0
    start = time.time()
    for batch_idx, batch in enumerate(loader):
        captions = batch['captions'].to(DEVICE)
        image_ids = batch['image_ids']
        batch_images = []
        for b in range(captions.size(0)):
            story_imgs = []
            for i in range(5):
                img = fetch_flickr_image(image_ids[i][b], transform)
                story_imgs.append(img)
            batch_images.append(torch.stack(story_imgs))
        images = torch.stack(batch_images).to(DEVICE)
        outputs = model(images, captions)
        loss = 0
        for i in range(5):
            pred = outputs[:, i].reshape(-1, len(vocab))
            targ = captions[:, i].reshape(-1)
            loss += criterion(pred, targ)
        loss /= 5
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start
            print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")
    return total_loss / len(loader)

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
best_loss = float('inf')
for epoch in range(1, NUM_EPOCHS + 1):
    loss = train_epoch(epoch)
    print(f"\n✅poch {epoch} complete | Avg Loss: {loss:.4f}")
    if loss < best_loss:
        best_loss = loss
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'vocab': vocab, 'loss': loss}, CHECKPOINT)
        print(f"💾 Checkpoint saved (best loss: {best_loss:.4f})")

print("\n🎉 Training complete!")
print(f"Best loss: {best_loss:.4f}")
