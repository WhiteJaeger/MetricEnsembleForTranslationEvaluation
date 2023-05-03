from pathlib import Path

import torch

from nn_training import MyNetwork

checkpoint = None
model_path = Path(__file__).parent.joinpath('models'). \
    joinpath(f'metric_ensemble_{checkpoint}.pt')

net: MyNetwork = torch.load(model_path)

print(float(net("The slow black plane flies over the big city.",
                "The fast brown fox jumps over the lazy dog.")[0]))

print(net("I am going shopping, would you come with me?",
          "The fast brown fox jumps over the lazy dog."))

print(net("The fast brown fox jumps over the lazy dog.",
          "The fast brown fox jumps over the lazy dog."))

print(net("x x x x x x x x x x x x x x.",
          "The fast brown fox jumps over the lazy dog."))
