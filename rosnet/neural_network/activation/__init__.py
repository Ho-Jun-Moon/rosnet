import numpy as np

from ._classes import (
    LINEAR,
    SIGMOID,
    RELU
)

def get_activation(name):
  activations = [RELU, SIGMOID, LINEAR]
  for activation in activations:
    if name.upper() == activation.__name__:
      return activation