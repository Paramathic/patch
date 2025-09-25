#!/bin/bash

mkdir /workspace/data

pip install --upgrade pip

pip install --no-cache-dir transformers==4.50.0 accelerate==0.34.2 datasets==3.0.1 SentencePiece wandb tqdm ninja tensorboardx==2.6 pulp timm einops nltk matplotlib seaborn
pip install flash-attn --no-build-isolation
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git

