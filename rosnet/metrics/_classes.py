import numpy as np
from ..decorator import check_vector

class MAE:
  def predict(self, y_predict, y):
    result = np.mean(np.abs(y_predict - y))
    return result
  
  def __str__(self):
    return 'mae'

class MSE:
  @check_vector
  def predict(self, y_predict, y):
    result = np.square(y_predict - y)
    result = np.mean(result)
    return result

  @check_vector
  def d(self, y_predict, y):
    diff = y_predict - y
    shape = diff.shape
    result = 1
    result = result * np.ones(shape)/np.prod(shape)
    result = result * 2 * diff

    return result

  def __str__(self):
    return 'mse'

class BINARY_CROSS_ENTROPY:
  @check_vector
  def predict(self, y_predict, y):
    H = self.get_H(y_predict, y)
    result = np.mean(H)

    return result

  def relu(self, X):
    return np.maximum(X,0)

  def transform(self, X):
    # input : 변환하고자하는 값
    # output[1] : 변환된 값
    # 시그모이드 값 대입하는 함수
    return np.exp(-self.relu(-X))/(1.0 + np.exp(-np.abs(X)))

  def derv_H_sigmoid(self, logit, z): # z는 참 레이블
    return -z + self.transform(logit)

  def get_H(self, X, z):
    return self.relu(X) - X * z + np.log(1 + np.exp(-np.abs(X)))
  
  @check_vector
  def d(self, y_predict, y):
    result = 1.0 / np.prod(y_predict.shape)
    result = result * self.derv_H_sigmoid(y_predict, y)    

    return result

  def __str__(self):
    return 'BINARY_CROSS_ENTROPY'

class ACCURACY:
  @check_vector
  def predict(self, y_predict, y):
    est_yes = np.greater(y_predict, 0.5)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    result = self.safe_div(tp+tn, tp+tn+fp+fn)
    return result

  def safe_div(self, p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20: return np.sign(p)
    return p / q

  def __str__(self):
    return 'ACCURACY'


class PRECISION:
  @check_vector
  def predict(self, y_predict, y):
    est_yes = np.greater(y_predict, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    result = self.safe_div(tp, tp+fp)
    return result

  def safe_div(self, p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20: return np.sign(p)
    return p / q

  def __str__(self):
    return 'PRECISION'

class RECALL:
  @check_vector
  def predict(self, y_predict, y):
    est_yes = np.greater(y_predict, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    result = self.safe_div(tp, tp+fn)
    return result

  def safe_div(self, p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20: return np.sign(p)
    return p / q

  def __str__(self):
    return 'RECALL'

class F1:
  @check_vector
  def predict(self, y_predict, y):
    est_yes = np.greater(y_predict, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    recall = self.safe_div(tp, tp+fn)
    precision = self.safe_div(tp, tp+fp)
    result = 2 * self.safe_div(recall*precision, recall+precision)
    return result

  def safe_div(self, p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20: return np.sign(p)
    return p / q

  def __str__(self):
    return 'F1'

class BINARY_CROSS_ENTROPY_WEIGHT:
  def __init__(self):
    self.weight = None
  
  @check_vector
  def predict(self, y_predict, y):
    if self.weight is None:
      self.weight = np.ones(y.shape) / len(y)

    H = self.get_H(y_predict, y)
    result = np.sum(H * self.weight)

    return result

  def set_weight(self, w):
    if len(w.shape) == 1:
      w = w.reshape(-1,1)
      
    self.weight = w

  def relu(self, X):
    return np.maximum(X,0)

  def transform(self, X):
    # input : 변환하고자하는 값
    # output[1] : 변환된 값
    # 시그모이드 값 대입하는 함수
    return np.exp(-self.relu(X))/(1.0 + np.exp(-np.abs(X)))

  def derv_H_sigmoid(self, logit, z): # z는 참 레이블
    return -z + self.transform(logit)

  def get_H(self, X, z):
    return self.relu(X) - X * z + np.log(1 + np.exp(-np.abs(X)))
  
  @check_vector
  def d(self, y_predict, y):
    result = self.weight
    result = result * self.derv_H_sigmoid(y_predict, y)    

    return result

  def __str__(self):
    return 'BINARY_CROSS_ENTROPY_WEIGHT'

