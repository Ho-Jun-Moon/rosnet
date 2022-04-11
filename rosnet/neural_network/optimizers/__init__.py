import numpy as np

from ._gradient_descent import (
  GD
)
from ._adam import ADAM

def get_optimizer(name):
  optimizers = [GD, ADAM]
  for optimizer in optimizers:
    if name.upper() == optimizer.__name__:
      return optimizer