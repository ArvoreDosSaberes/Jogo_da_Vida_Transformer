from __future__ import annotations

import numpy as np
import torch

from model import NeighborhoodTransformer, get_device, predict_batch, train_step


class GameOfLife:
    def __init__(self, rows: int = 50, cols: int = 80, p_alive: float = 0.2):
        self.rows = rows
        self.cols = cols
        self.grid = (np.random.rand(rows, cols) < p_alive).astype(np.int64)

    def randomize(self, p_alive: float = 0.2):
        self.grid = (np.random.rand(self.rows, self.cols) < p_alive).astype(np.int64)

    def step_classic(self):
        g = self.grid
        # Count neighbors using convolution-like sums via roll
        neighbors = sum(
            np.roll(np.roll(g, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if not (i == 0 and j == 0)
        )
        # Apply rules
        born = (g == 0) & (neighbors == 3)
        survive = (g == 1) & ((neighbors == 2) | (neighbors == 3))
        self.grid = np.where(born | survive, 1, 0).astype(np.int64)

    def _extract_neighborhoods(self, grid: np.ndarray) -> np.ndarray:
        """Return a (rows*cols, 9) array of 3x3 neighborhoods (wrap-around)."""
        r, c = grid.shape
        neighs = np.zeros((r * c, 9), dtype=np.int64)
        idx = 0
        for i in range(r):
            for j in range(c):
                # Collect 3x3 neighborhood with wrap-around
                block = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        block.append(grid[(i + di) % r, (j + dj) % c])
                neighs[idx] = np.array(block, dtype=np.int64)
                idx += 1
        return neighs

    def step_ai(self, model: NeighborhoodTransformer, device: torch.device | None = None, batch_size: int = 8192):
        if device is None:
            device = get_device()
        g = self.grid
        neighs = self._extract_neighborhoods(g)  # (N,9)
        N = neighs.shape[0]
        preds = np.zeros((N,), dtype=np.int64)
        # Batch to avoid GPU OOM
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch = torch.from_numpy(neighs[start:end]).long()
            preds[start:end] = predict_batch(model, batch, device).numpy()
        self.grid = preds.reshape(g.shape)

    def sample_training_batch(self, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a batch of neighborhoods and classic-rule targets from current grid."""
        # Build neighborhoods
        neighs = self._extract_neighborhoods(self.grid)
        # Targets from classic next state
        g0 = self.grid.copy()
        # Compute one classic step to get targets
        self.step_classic()
        targets_grid = self.grid.copy()
        # Restore original grid as we don't actually want to advance here
        self.grid = g0
        targets = targets_grid.reshape(-1)
        # Randomly sample batch_size examples
        idx = np.random.choice(neighs.shape[0], size=min(batch_size, neighs.shape[0]), replace=False)
        batch_neigh = torch.from_numpy(neighs[idx]).long()
        batch_targets = torch.from_numpy(targets[idx]).long()
        return batch_neigh, batch_targets


def create_model_and_optim(d_model: int = 32, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 64, lr: float = 1e-3):
    model = NeighborhoodTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
    device = get_device()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optim, device
