import numpy as np
from ._threshold import Threshold

class TreeNode:
  def __init__(self, parent = None, left = None, right = None):
    if parent is not None:
      self.is_root = True
    else:
      self.is_root = False

    if left is None:
      self.left_ = left
    if right is None:
      self.right_ = right
    self.step_ = 0.1
    self.continue_limit_ = 4
    self.cost_ = 1
  def add_left(self, left):
    self.left_ = left
  
  def add_right(self, right):
    self.right_ = right
  
  def fit(self, x, y, weights = None):
    except_col = range(len(y.shape))
    d = 0
    for col_max in x.shape:
      if d in except_col:
        d += 1
        continue
      
      for col in range(col_max):
        axis = list(range(len(x.shape)))
        axis.pop(d)
        axis.insert(0, d)
        column = np.transpose(x, axis)[col]
        unique_ = np.unique(column)

        # unique가 discrete로 취급하는게 편한 경우
        if len(unique_) < self.continue_limit_:
          for uni in unique_:
            new_threshold = Threshold(uni, axis, col, False)
            condition = column == uni
            new_cost = new_threshold.get_cost(y, condition)
            if self.cost_ > new_cost:
              self.cost_ = new_cost
              self.threshold_ = new_threshold

        # unique가 conitnue로 취급하는게 편한 경우
        else:
          for uni in unique_:
            if np.isnan(uni):
              new_threshold = Threshold(uni, axis, col, False)
              condition = np.isnan(column)
              new_cost = new_threshold.get_cost(y, condition)
              if self.cost_ > new_cost:
                self.cost_ = new_cost
                self.threshold_ = new_threshold

          for q in np.arange(self.step_, 1, self.step_):
            uni = np.quantile(column, q)
            new_threshold = Threshold(uni, axis, col, True)
            condition = column < uni
            new_cost = new_threshold.get_cost(y, condition)
            if self.cost_ > new_cost:
              self.cost_ = new_cost
              self.threshold_ = new_threshold

    
      d += 1

  def predict(self, x, y=None):
    if y is None:
      return self.threshold_.divide(x)
    else:
      return self.threshold_.divide(x, y)

  def __str__(self):
    return f"[Threshold_axis] : {self.threshold_.axis_[0]} [T_col] : {self.threshold_.col_} [T_unit] : {self.threshold_.unit_:.3f} [cost] : {self.cost_:.3f}"
