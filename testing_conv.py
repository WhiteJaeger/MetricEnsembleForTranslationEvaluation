from pathlib import Path

import torch

from training_conv import EMEQT

model_name = 'metric_ensemble_conv_2023-05-05_23:53:42.pt'
model_path = Path(__file__).parent.joinpath('models'). \
    joinpath(model_name)

net: EMEQT = torch.load(model_path)

net.eval()

print(net)

print(f'TOTAL # OF PARAMETERS: {sum(p.numel() for p in net.parameters())}')

print(net("The slow black plane flies over the big city.",
          "The fast brown fox jumps over the lazy dog."))

print(net("I am going shopping, would you come with me?",
          "The fast brown fox jumps over the lazy dog."))

print(net("The fast brown fox jumps over the lazy dog.",
          "The fast brown fox jumps over the lazy dog."))

print(net("x x x x x x x x x x x x x x.",
          "The fast brown fox jumps over the lazy dog."))
