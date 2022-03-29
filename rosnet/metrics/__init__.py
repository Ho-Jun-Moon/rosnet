from ._classes import (
    MAE,
    MSE,
    BINARY_CROSS_ENTROPY,
    ACCURACY,
    RECALL,
    PRECISION,
    F1
)

def get_metrics(name):
  metrics = [MAE, MSE, BINARY_CROSS_ENTROPY, ACCURACY, PRECISION, RECALL, F1]
  for metric in metrics:
    if name.upper() == metric.__name__:
      return metric