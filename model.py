# model.py
import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, max_len=128, embed_dim=128, heads=4, layers=4):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, time)
        b, t = x.size()
        # Positions-Tensor auf das gleiche Device wie x legen
        positions = torch.arange(0, t, device=x.device)
        x = self.embed(x) + self.pos(positions)
        x = self.transformer(x)
        return self.fc(x)
