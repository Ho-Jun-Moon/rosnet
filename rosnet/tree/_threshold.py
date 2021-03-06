import numpy as np

class Threshold:
  def __init__(self, unit, axis, col, is_continue):
    self.unit_ = unit
    self.axis_ = axis
    self.is_continue_ = is_continue
    self.col_ = col
  
  def get_cost(self, y, condition):
    cost_true = 1
    cost_false = 1
    y_true = y[condition]
    y_false = y[~condition]
    
    # 이 부분부터 y가 벡터로 고정이 되어 있다.
    tot_true = y_true.shape[0]
    tot_false = y_false.shape[0]
    if tot_true * tot_false == 0: 
      return 2
    tot = tot_true + tot_false

    for uni in np.unique(y):
      count_true = np.sum(y_true == uni)
      count_false = np.sum(y_false == uni)
      cost_true -= np.power(count_true/tot_true, 2)
      cost_false -= np.power(count_false/tot_false, 2)

    cost = tot_true/tot * cost_true + tot_false/tot * cost_false
    return cost

  def divide(self, x, y = None):
    column = np.transpose(x, self.axis_)[self.col_]
    if y is not None:
      if self.is_continue_:
        condition = column < self.unit_
        return x[condition], x[~condition], y[condition], y[~condition]

      elif np.isnan(self.unit_):
        condition = np.isnan(column)
        return x[condition], x[~condition], y[condition], y[~condition]
      
      else:
        condition = column == self.unit_
        return x[condition], x[~condition], y[condition], y[~condition]
    
    else:
      if self.is_continue_:
        condition = column < self.unit_
        return x[condition], x[~condition]

      elif np.isnan(self.unit_):
        condition = np.isnan(column)
        return x[condition], x[~condition]
      
      else:
        condition = column == self.unit_
        return x[condition], x[~condition]