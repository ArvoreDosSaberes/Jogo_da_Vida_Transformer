import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NeighborhoodTransformer(nn.Module):
    """
    A tiny Transformer that takes a 3x3 neighborhood (flattened to length=9)
    of binary states (0/1) and predicts the next state of the center cell.

    Input: batch_size x 9 (ints 0/1)
    Output: batch_size x 2 logits (for classes 0 and 1)
    """

    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64):
        super().__init__()
        self.token_emb = nn.Embedding(2, d_model)  # tokens are 0 or 1
        self.pos_emb = nn.Parameter(torch.zeros(1, 9, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9) with ints {0,1}
        h = self.token_emb(x)  # (B, 9, d_model)
        h = h + self.pos_emb
        h = self.encoder(h)  # (B, 9, d_model)
        # Use the embedding at the center position (index 4) or mean-pool
        center = h[:, 4, :]  # (B, d_model)
        logits = self.head(center)  # (B, 2)
        return logits


@torch.no_grad()
def predict_batch(model: NeighborhoodTransformer, batch_neigh: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Predict binary next-states for a batch of neighborhoods.

    batch_neigh: (B, 9) LongTensor of {0,1}
    returns: (B,) LongTensor of {0,1}
    """
    model.eval()
    batch_neigh = batch_neigh.to(device)
    logits = model(batch_neigh)
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).to(torch.long)
    return pred.cpu()


def train_step(model: NeighborhoodTransformer, optimizer: torch.optim.Optimizer, batch_neigh: torch.Tensor, batch_targets: torch.Tensor, device: torch.device) -> float:
    """One optimization step with cross-entropy loss.

    batch_neigh: (B, 9) LongTensor {0,1}
    batch_targets: (B,) LongTensor {0,1}
    returns: loss value (float)
    """
    model.train()
    batch_neigh = batch_neigh.to(device)
    batch_targets = batch_targets.to(device)

    logits = model(batch_neigh)
    loss = F.cross_entropy(logits, batch_targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())
