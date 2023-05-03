from pathlib import Path

import torch

from training import MyNetwork


checkpoint = '2023-05-03_21:50:47'
model_path = Path(__file__).parent.joinpath('models'). \
    joinpath(f'metric_ensemble_{checkpoint}.pt')

net: MyNetwork = torch.load(model_path)

print(net("The slow black plane flies over the big city.",
          "The fast brown fox jumps over the lazy dog."))

print(net("I am going shopping, would you come with me?",
          "The fast brown fox jumps over the lazy dog."))

print(net("The fast brown fox jumps over the lazy dog.",
          "The fast brown fox jumps over the lazy dog."))

print(net("x x x x x x x x x x x x x x.",
          "The fast brown fox jumps over the lazy dog."))
