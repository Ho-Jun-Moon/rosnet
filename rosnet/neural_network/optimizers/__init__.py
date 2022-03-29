import numpy as np

from ._gradient_descent import (
  GD
)

def get_optimizer(name):
  optimizers = [GD]
  for optimizer in optimizers:
    if name.upper() == optimizer.__name__:
      return optimizer