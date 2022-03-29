import numpy as np

from ._classes import (
    MAE,
    MSE,
    BINARY_CROSS_ENTROPY,
    ACCURACY,
    RECALL,
    PRECISION,
    F1,
    BINARY_CROSS_ENTROPY_WEIGHT
)

def get_metrics(name):
  metrics = [ MAE, 
              MSE, 
              BINARY_CROSS_ENTROPY, 
              ACCURACY, 
              PRECISION, 
              RECALL, 
              F1, 
              BINARY_CROSS_ENTROPY_WEIGHT]

  for metric in metrics:
    if name.upper() == metric.__name__:
      return metric