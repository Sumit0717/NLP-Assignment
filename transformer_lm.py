import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

torch.manual_seed(42)
random.seed(42)

sentences = [
    "i love natural language processing",
    "transformers are powerful deep learning models",
    "language models can generate text",
    "deep learning enables artificial intelligence",
    "natural language processing is fascinating",
    "machine learning is a part of artificial intelligence",
    "transformers use attention mechanisms",
    "attention allows models to focus on important words",
    "text generation is an important task",
    "neural networks learn patterns from data",
    "language understanding requires context",
    "deep neural networks power modern ai systems",
    "artificial intelligence is transforming industries",
    "language models predict the next word",
    "transformers improve language understanding"
]

words = list(set(" ".join(sentences).split()))
vocab = {word: i for i, word in enumerate(words)}
reverse_vocab = {i: word for word, i in vocab.items()}
vocab_size = len(vocab)

def encode(sentence):
    return torch.tensor([vocab[word] for word in sentence.split()], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tgt):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        tgt = tgt.permute(1, 0, 2)

        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0))
        output = self.decoder(tgt, tgt, tgt_mask=mask)

        output = output.permute(1, 0, 2)
        return self.fc_out(output)

model = TransformerLM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500

for epoch in range(epochs):
    total_loss = 0

    for sentence in sentences:
        data = encode(sentence)
        input_seq = data[:-1].unsqueeze(0)
        target_seq = data[1:].unsqueeze(0)

        optimizer.zero_grad()
        output = model(input_seq)

        loss = criterion(output.reshape(-1, vocab_size),
                         target_seq.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("\nTraining Complete\n")

def generate(model, start_text, max_len=8, temperature=0.9):
    model.eval()
    words = start_text.split()
    input_ids = torch.tensor([vocab[word] for word in words]).unsqueeze(0)

    for _ in range(max_len):
        output = model(input_ids)
        logits = output[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)
        next_word_id = torch.multinomial(probs, 1).item()

        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_word_id]])],
            dim=1
        )

    result = [reverse_vocab[i.item()] for i in input_ids[0]]
    return " ".join(result)

print(generate(model, "language"))
print(generate(model, "transformers"))
print(generate(model, "artificial intelligence"))
