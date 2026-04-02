import json
import os
from collections import defaultdict

# Path to our annotations
DATA_DIR = "shared/dataset/sis"
TRAIN_FILE = os.path.join(DATA_DIR, "train.story-in-sequence.json")

def load_vist(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def explore_dataset(data):
    annotations = data['annotations']
    print(f"Total story annotations: {len(annotations)}")
    
    # Group by story ID
    stories = defaultdict(list)
    for ann_group in annotations:
        for ann in ann_group:
            story_id = ann['story_id']
            stories[story_id].append(ann)
    
    print(f"Total unique stories: {len(stories)}")
    
    # Show a sample story
    sample_id = list(stories.keys())[0]
    sample = sorted(stories[sample_id], key=lambda x: x['worker_arranged_photo_order'])
    
    print(f"\n--- Sample Story ID: {sample_id} ---")
    for i, sentence in enumerate(sample):
        print(f"Photo {i+1}: {sentence['text']}")
        print(f"  Image ID: {sentence['photo_flickr_id']}")
    
    return stories

if __name__ == "__main__":
    print("Loading VIST training data...")
    data = load_vist(TRAIN_FILE)
    stories = explore_dataset(data)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Ready to train on {len(stories)} stories")
    