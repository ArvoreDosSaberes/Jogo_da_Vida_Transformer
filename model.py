"""
Módulo de IA: define um pequeno Transformer para prever o próximo estado da célula
central a partir do bairro 3x3 (total de 9 posições) do Jogo da Vida.

Ideia principal:
- Transformamos cada célula do bairro (0=vida ausente, 1=vida presente) em um
  token e aplicamos um Transformer Encoder para extrair relações entre as nove
  posições.
- A predição final (0 ou 1) é feita a partir do embedding da posição central
  (índice 4) usando uma pequena cabeça linear.

Este modelo é treinado com professor (as regras clássicas do Jogo da Vida),
minimizando a entropia cruzada entre o alvo (próximo estado clássico) e a
saída do Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Seleciona automaticamente o dispositivo de execução (GPU se disponível).

    - Se houver CUDA disponível, usa `cuda`.
    - Caso contrário, recorre à CPU.
    """
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class NeighborhoodTransformer(nn.Module):
    """
    Transformer compacto para o bairro 3x3 do Jogo da Vida.

    Entrada
    - Tensor de shape (B, 9) com inteiros {0,1} representando o bairro 3x3
      achatado (ordem linha-a-linha) onde o índice 4 é a célula central.

    Saída
    - Logits de shape (B, 2), correspondendo às classes {morta(0), viva(1)}
      para o próximo estado da célula central.

    Detalhes
    - `nn.Embedding(2, d_model)`: cada token 0/1 vira um vetor denso.
    - `pos_emb`: parâmetro de posição para distinguir as 9 posições do bairro.
    - `TransformerEncoder`: modela interações entre as posições do bairro.
    - `head`: normaliza e projeta para 2 logits.
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
        # x: (B, 9) com ints {0,1}
        h = self.token_emb(x)  # (B, 9, d_model)
        h = h + self.pos_emb
        h = self.encoder(h)  # (B, 9, d_model)
        # Estratégia: usar o embedding da posição central (índice 4)
        # Poderíamos também fazer uma média (mean-pooling) das 9 posições.
        center = h[:, 4, :]  # (B, d_model)
        logits = self.head(center)  # (B, 2)
        return logits


@torch.no_grad()
def predict_batch(model: NeighborhoodTransformer, batch_neigh: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Prediz o próximo estado (0/1) para um lote de bairros 3x3.

    Parâmetros
    - model: modelo Transformer já carregado no dispositivo.
    - batch_neigh: tensor (B, 9) com inteiros {0,1}.
    - device: dispositivo de execução (CPU/GPU).

    Retorno
    - Tensor (B,) com inteiros {0,1} correspondendo ao estado previsto.
    """
    model.eval()
    batch_neigh = batch_neigh.to(device)
    logits = model(batch_neigh)
    probs = F.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).to(torch.long)
    return pred.cpu()


def train_step(model: NeighborhoodTransformer, optimizer: torch.optim.Optimizer, batch_neigh: torch.Tensor, batch_targets: torch.Tensor, device: torch.device) -> float:
    """Executa um passo de treino com perda de entropia cruzada.

    Parâmetros
    - batch_neigh: (B, 9) LongTensor {0,1} com os bairros 3x3.
    - batch_targets: (B,) LongTensor {0,1} com o próximo estado (alvo clássico).

    Retorno
    - Valor escalar da loss (float) para monitoramento.
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
