"""
Interface gráfica (Tkinter) para o Jogo da Vida com dois modos de evolução:

- Clássico: aplica diretamente as regras de Conway.
- IA (Transformer): usa um pequeno modelo que tenta imitar a regra clássica
  a partir do bairro 3x3. Pode ser treinado online pela própria GUI.

Controles principais:
- Start/Stop: liga/desliga a simulação automática (timer com `after`).
- Step: avança uma única iteração.
- Randomize: randomiza o grid atual.
- Mode (Classic/AI): alterna o modo de evolução.
- Train AI (xN iters): executa N iterações rápidas de treino supervisionado.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np

from game import GameOfLife, create_model_and_optim
from model import train_step


class GameGUI:
    """Componente principal da GUI e loop de interação.

    Parâmetros
    - root: janela raiz do Tkinter.
    - rows, cols: dimensões do grid exibido.
    - cell_size: tamanho (px) de cada célula desenhada no Canvas.
    """
    def __init__(self, root: tk.Tk, rows: int = 50, cols: int = 80, cell_size: int = 10):
        self.root = root
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.running = False
        self.mode = tk.StringVar(value="Classic")  # Classic or AI

        # Game and AI
        self.game = GameOfLife(rows=rows, cols=cols, p_alive=0.25)
        self.model, self.optim, self.device = create_model_and_optim()

        # UI
        self.root.title("Jogo da Vida - Classic / AI (Transformer)")
        self._build_widgets()
        self._draw()

    def _build_widgets(self):
        """Cria os botões de controle e o Canvas de desenho."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.start_btn = ttk.Button(control_frame, text="Start", command=self.toggle_run)
        self.start_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.step_btn = ttk.Button(control_frame, text="Step", command=self.step_once)
        self.step_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.rand_btn = ttk.Button(control_frame, text="Randomize", command=self.randomize)
        self.rand_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.mode_btn = ttk.Button(control_frame, textvariable=self.mode, command=self.toggle_mode)
        self.mode_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.train_btn = ttk.Button(control_frame, text="Train AI (x200 iters)", command=lambda: self.train_ai(200))
        self.train_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.loss_var = tk.StringVar(value="loss: -")
        self.loss_lbl = ttk.Label(control_frame, textvariable=self.loss_var)
        self.loss_lbl.pack(side=tk.LEFT, padx=8)

        canvas_w = self.cols * self.cell_size
        canvas_h = self.rows * self.cell_size
        self.canvas = tk.Canvas(self.root, width=canvas_w, height=canvas_h, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.canvas.bind("<Button-1>", self.on_click)

    def toggle_run(self):
        """Inicia/para o loop automático de evolução (timer com 50ms)."""
        self.running = not self.running
        self.start_btn.config(text="Stop" if self.running else "Start")
        if self.running:
            self._tick()

    def step_once(self):
        """Avança uma única iteração e redesenha a tela."""
        self._step_logic()
        self._draw()

    def randomize(self):
        """Randomiza o grid e redesenha."""
        self.game.randomize(p_alive=0.25)
        self._draw()

    def toggle_mode(self):
        """Alterna entre os modos Clássico e IA (texto do botão também muda)."""
        current = self.mode.get()
        self.mode.set("AI" if current == "Classic" else "Classic")

    def on_click(self, event):
        """Inverte manualmente o estado da célula clicada (edição do grid)."""
        i = event.y // self.cell_size
        j = event.x // self.cell_size
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.game.grid[i, j] = 1 - self.game.grid[i, j]
            self._draw()

    def _tick(self):
        """Callback periódico enquanto `running=True`.

        Usa `after(50, self._tick)` para agendar o próximo passo, evitando
        travar a GUI (não bloqueia o thread principal do Tk).
        """
        if not self.running:
            return
        self._step_logic()
        self._draw()
        self.root.after(50, self._tick)

    def _step_logic(self):
        """Seleciona a lógica de evolução de acordo com o modo atual."""
        if self.mode.get() == "Classic":
            self.game.step_classic()
        else:
            # AI mode
            self.game.step_ai(self.model, device=self.device)

    def _draw(self):
        """Desenha o grid no Canvas.

        Otimização simples: ao invés de desenhar cada célula individualmente,
        agrupamos sequências contíguas de células vivas por linha em um único
        retângulo, reduzindo o número de objetos no Canvas.
        """
        self.canvas.delete("all")
        g = self.game.grid
        cs = self.cell_size
        for i in range(self.rows):
            y0 = i * cs
            y1 = y0 + cs
            # Fast path: draw rectangles per run of alive cells per row
            row = g[i]
            j = 0
            while j < self.cols:
                if row[j] == 1:
                    start = j
                    while j < self.cols and row[j] == 1:
                        j += 1
                    end = j
                    x0 = start * cs
                    x1 = end * cs
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="", fill="#222")
                else:
                    j += 1

    def train_ai(self, iters: int = 200, batch_size: int = 4096):
        # Executa algumas iterações de treino online usando o clássico como professor.
        # Observação: isso é propositalmente simples e rápido para fins didáticos.
        losses = []
        for _ in range(iters):
            batch_neigh, batch_targets = self.game.sample_training_batch(batch_size=batch_size)
            loss = train_step(self.model, self.optim, batch_neigh, batch_targets, device=self.device)
            losses.append(loss)
        if losses:
            self.loss_var.set(f"loss: {np.mean(losses):.4f}")


def run_app():
    """Função de entrada que instancia a janela raiz e a `GameGUI`."""
    root = tk.Tk()
    # Slightly smaller default grid for speed in Python/Tk
    app = GameGUI(root, rows=50, cols=80, cell_size=10)
    root.mainloop()
