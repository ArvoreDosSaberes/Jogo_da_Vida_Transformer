# Jogo da Vida com Transformer (PyTorch + Tkinter)

Este projeto implementa o Jogo da Vida (Conway) com uma GUI em Tkinter e um modelo de IA baseado em Transformer (PyTorch) que aprende a aproximar a regra do jogo e pode dirigir a simulação.

## Requisitos

- Python 3.10+
- Linux (testado)
- Dependências em `requirements.txt`

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

```bash
python main.py
```

## Como funciona

- Modo Clássico: aplica a regra tradicional do Jogo da Vida.
- Modo IA: utiliza um pequeno Transformer para prever o próximo estado de cada célula a partir do seu bairro 3x3.
- Treinar IA: treina online o Transformer usando a regra clássica como professor, a partir de amostras de bairros 3x3 do grid atual.

## Controles (GUI)

- Start/Stop: inicia/pausa a simulação.
- Step: avança um passo.
- Randomize: randomiza o grid.
- Mode: alterna entre "Classic" e "AI".
- Train AI: executa algumas iterações de treino rápido.

## Observações

- Se houver GPU disponível (CUDA), o modelo tentará utilizá-la.
- O treinamento online é simples e serve para demonstração; para melhor qualidade, rode mais iterações e varie amostras.
