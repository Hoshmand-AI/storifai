import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from PIL import Image
import requests
from io import BytesIO

# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    """Converts words to numbers and back."""
    
    def __init__(self, freq_threshold=3):
        self.freq_threshold = freq_threshold
        # Special tokens every vocabulary needs
        self.word2idx = {
            "<PAD>": 0,   # padding — makes all sentences same length
            "<SOS>": 1,   # start of sentence
            "<EOS>": 2,   # end of sentence
            "<UNK>": 3,   # unknown word
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_freq = Counter()

    def build(self, stories):
        """Build vocabulary from all stories."""
        print("Building vocabulary...")
        # Count every word in every story
        for story in stories:
            for sentence in story:
                for word in sentence.lower().split():
                    self.word_freq[word] += 1

        # Only keep words that appear at least freq_threshold times
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)} words")
        return self

    def encode(self, sentence):
        """Convert sentence to list of numbers."""
        tokens = sentence.lower().split()
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

    def decode(self, indices):
        """Convert list of numbers back to sentence."""
        words = [self.idx2word.get(i, "<UNK>") for i in indices]
        # Stop at <EOS>
        if "<EOS>" in words:
            words = words[:words.index("<EOS>")]
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class VISTDataset(Dataset):
    """PyTorch Dataset for VIST stories."""

    def __init__(self, json_path, vocab=None, max_len=30, transform=None):
        self.max_len = max_len
        self.transform = transform
        self.stories = self._load(json_path)
        
        # Build vocabulary if not provided
        if vocab is None:
            all_sentences = [s for story in self.stories 
                           for s in story['sentences']]
            self.vocab = Vocabulary().build(
                [story['sentences'] for story in self.stories]
            )
        else:
            self.vocab = vocab

        print(f"Dataset loaded: {len(self.stories)} stories")

    def _load(self, json_path):
        """Load and parse VIST JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Group annotations by story_id
        from collections import defaultdict
        grouped = defaultdict(list)
        for ann_group in data['annotations']:
            for ann in ann_group:
                grouped[ann['story_id']].append(ann)

        # Build clean story list
        stories = []
        for story_id, anns in grouped.items():
            anns = sorted(anns, key=lambda x: x['worker_arranged_photo_order'])
            if len(anns) == 5:  # Only complete 5-photo stories
                stories.append({
                    'story_id': story_id,
                    'image_ids': [a['photo_flickr_id'] for a in anns],
                    'sentences': [a['text'] for a in anns],
                    'urls': [a.get('url_o', '') for a in anns]
                })
        return stories

    def _encode_story(self, sentences):
        """Convert 5 sentences into padded token tensors."""
        encoded = []
        for sent in sentences:
            tokens = ([self.vocab.word2idx["<SOS>"]] +
                     self.vocab.encode(sent) +
                     [self.vocab.word2idx["<EOS>"]])
            # Pad or truncate to max_len
            if len(tokens) < self.max_len:
                tokens += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(tokens))
            else:
                tokens = tokens[:self.max_len]
            encoded.append(torch.tensor(tokens, dtype=torch.long))
        return torch.stack(encoded)  # Shape: [5, max_len]

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        # Encode the 5 sentences
        captions = self._encode_story(story['sentences'])
        return {
            'story_id': story['story_id'],
            'image_ids': story['image_ids'],
            'captions': captions,       # [5, max_len] tensor
            'sentences': story['sentences']  # raw text
        }


# ─── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_JSON = "shared/dataset/sis/train.story-in-sequence.json"
    
    print("Loading dataset...")
    dataset = VISTDataset(TRAIN_JSON)
    
    # Test one sample
    sample = dataset[0]
    print(f"\nStory ID: {sample['story_id']}")
    print(f"Image IDs: {sample['image_ids']}")
    print(f"Captions tensor shape: {sample['captions'].shape}")
    print(f"\nSentences:")
    for i, sent in enumerate(sample['sentences']):
        print(f"  {i+1}: {sent}")
    
    print(f"\nEncoded caption 1: {sample['captions'][0]}")
    print(f"Decoded back: {dataset.vocab.decode(sample['captions'][0].tolist())}")
    
    print(f"\n✅ Dataloader working!")
    print(f"Vocabulary size: {len(dataset.vocab)}")
    print(f"Total stories: {len(dataset)}")