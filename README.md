[![CI](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/actions/workflows/ci.yml/badge.svg)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/actions/workflows/ci.yml)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=ArvoreDosSaberes.Jogo_da_Vida_Transformer)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
![Language: Portuguese](https://img.shields.io/badge/Language-Portuguese-brightgreen.svg)
[![Language-C](https://img.shields.io/badge/language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![CMake](https://img.shields.io/badge/build-CMake-informational.svg)](https://cmake.org/)
[![Raylib](https://img.shields.io/badge/graphics-raylib-2ea44f.svg)](https://www.raylib.com/)
[![Issues](https://img.shields.io/github/issues/ArvoreDosSaberes/Jogo_da_Vida_Transformer.svg)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/issues)
[![Stars](https://img.shields.io/github/stars/ArvoreDosSaberes/Jogo_da_Vida_Transformer.svg)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/stargazers)
[![Forks](https://img.shields.io/github/forks/ArvoreDosSaberes/Jogo_da_Vida_Transformer.svg)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/network/members)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)
[![Watchers](https://img.shields.io/github/watchers/ArvoreDosSaberes/Jogo_da_Vida_Transformer)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/watchers)
[![Last Commit](https://img.shields.io/github/last-commit/ArvoreDosSaberes/Jogo_da_Vida_Transformer)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/commits)
[![Contributors](https://img.shields.io/github/contributors/ArvoreDosSaberes/Jogo_da_Vida_Transformer)](https://github.com/ArvoreDosSaberes/Jogo_da_Vida_Transformer/graphs/contributors)


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
