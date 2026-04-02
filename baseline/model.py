import torch
import torch.nn as nn
import torchvision.models as models


# ─── Image Encoder ────────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """
    ResNet-50 pretrained on ImageNet.
    Takes a photo → outputs a 256-dim vector (image representation).
    """

    def __init__(self, embed_size=256):
        super(ImageEncoder, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final classification layer
        # We don't want to classify images — we want features
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Add our own projection layer: 2048 → embed_size
        self.projection = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

        # Freeze ResNet weights — only train our projection layer
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        """
        images: [batch, 3, 224, 224]
        returns: [batch, embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images)          # [batch, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch, 2048]
        features = self.projection(features)         # [batch, embed_size]
        features = self.bn(features)                 # normalize
        return features


# ─── Story Context ────────────────────────────────────────────────────────────
class StoryContext(nn.Module):
    """
    Combines 5 image representations into one story context vector.
    Simple approach: average all 5 image features.
    """

    def __init__(self):
        super(StoryContext, self).__init__()

    def forward(self, image_features):
        """
        image_features: [batch, 5, embed_size]
        returns: [batch, embed_size]
        """
        # Average the 5 image representations
        context = image_features.mean(dim=1)  # [batch, embed_size]
        return context


# ─── LSTM Decoder ─────────────────────────────────────────────────────────────
class LSTMDecoder(nn.Module):
    """
    Generates a story sentence by sentence.
    Takes image context → produces words one at a time.
    """

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()

        # Word embedding: number → vector
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # LSTM: the core of the decoder
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )

        # Output layer: hidden state → word probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, context, captions):
        """
        context:  [batch, embed_size] — image context vector
        captions: [batch, seq_len]    — target story tokens
        returns:  [batch, seq_len, vocab_size] — word predictions
        """
        # Embed the caption tokens
        embeddings = self.dropout(self.embedding(captions))  # [batch, seq_len, embed_size]

        # Prepend context as the first input (teacher forcing)
        context = context.unsqueeze(1)  # [batch, 1, embed_size]
        inputs = torch.cat([context, embeddings[:, :-1, :]], dim=1)  # [batch, seq_len, embed_size]

        # Run through LSTM
        outputs, _ = self.lstm(inputs)  # [batch, seq_len, hidden_size]

        # Project to vocabulary
        predictions = self.fc(outputs)  # [batch, seq_len, vocab_size]
        return predictions

    def generate(self, context, vocab, max_len=30, device='cpu'):
        """
        Generate a story sentence given image context.
        Uses greedy decoding — picks highest probability word each step.
        """
        self.eval()
        with torch.no_grad():
            # Start with <SOS> token
            word_idx = torch.tensor([[vocab.word2idx["<SOS>"]]], 
                                     device=device)
            hidden = None
            sentence = []

            # Use context as first input
            inp = context.unsqueeze(1)  # [1, 1, embed_size]
            output, hidden = self.lstm(inp, hidden)
            pred = self.fc(output.squeeze(1))
            word_idx = pred.argmax(dim=-1, keepdim=True)

            for _ in range(max_len):
                emb = self.embedding(word_idx)          # [1, 1, embed_size]
                output, hidden = self.lstm(emb, hidden) # [1, 1, hidden_size]
                pred = self.fc(output.squeeze(1))       # [1, vocab_size]
                word_idx = pred.argmax(dim=-1, keepdim=True)

                word = vocab.idx2word.get(word_idx.item(), "<UNK>")
                if word == "<EOS>":
                    break
                if word not in ["<PAD>", "<SOS>", "<UNK>"]:
                    sentence.append(word)

        return " ".join(sentence)


# ─── Full Baseline Model ──────────────────────────────────────────────────────
class StorifaiBaseline(nn.Module):
    """
    Complete baseline model:
    5 photos → ResNet-50 → average → LSTM → story
    """

    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=1):
        super(StorifaiBaseline, self).__init__()

        self.encoder = ImageEncoder(embed_size)
        self.context = StoryContext()
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        """
        images:   [batch, 5, 3, 224, 224] — 5 photos per story
        captions: [batch, 5, seq_len]     — 5 sentences per story
        returns:  predictions for each sentence
        """
        batch_size = images.size(0)

        # Encode all 5 images
        image_features = []
        for i in range(5):
            feat = self.encoder(images[:, i])  # [batch, embed_size]
            image_features.append(feat)
        image_features = torch.stack(image_features, dim=1)  # [batch, 5, embed_size]

        # Get story context
        context = self.context(image_features)  # [batch, embed_size]

        # Generate predictions for all 5 sentences
        all_predictions = []
        for i in range(5):
            preds = self.decoder(context, captions[:, i])  # [batch, seq_len, vocab_size]
            all_predictions.append(preds)

        return torch.stack(all_predictions, dim=1)  # [batch, 5, seq_len, vocab_size]

    def generate_story(self, images, vocab, max_len=30, device='cpu'):
        """Generate a complete 5-sentence story from 5 photos."""
        batch_size = images.size(0)

        # Encode images
        image_features = []
        for i in range(5):
            feat = self.encoder(images[:, i])
            image_features.append(feat)
        image_features = torch.stack(image_features, dim=1)

        # Get context
        context = self.context(image_features)

        # Generate each sentence
        story = []
        for i in range(5):
            sentence = self.decoder.generate(
                context[0:1], vocab, max_len, device
            )
            story.append(f"Photo {i+1}: {sentence}")

        return story


# ─── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Storifai Baseline Model...")

    # Dummy vocabulary size
    VOCAB_SIZE = 12979
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    BATCH_SIZE = 2

    # Create model
    model = StorifaiBaseline(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Test forward pass with dummy data
    dummy_images = torch.randn(BATCH_SIZE, 5, 3, 224, 224)
    dummy_captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 5, 30))

    print(f"\nInput images shape:   {dummy_images.shape}")
    print(f"Input captions shape: {dummy_captions.shape}")

    output = model(dummy_images, dummy_captions)
    print(f"Output shape:         {output.shape}")
    print(f"Expected:             [2, 5, 30, 12979]")

    print("\n✅ Model architecture working!")
    print("Ready for training.")