import numpy as np

class MAE:
  def predict(self, y_predict, y):
    result = np.mean(np.abs(y_predict - y))
    return result
  
  def __str__(self):
    return 'mae'

class MSE:
  def predict(self, y_predict, y):
    result = np.square(y_predict - y)
    result = np.mean(result)
    return result

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
    return np.exp(-self.relu(X))/(1.0 + np.exp(-np.abs(X)))

  def derv_H_sigmoid(self, logit, z): # z는 참 레이블
    return -z + self.transform(logit)

  def get_H(self, X, z):
    return self.relu(X) - X * z + np.log(1 + np.exp(-np.abs(X)))
  
  def d(self, y_predict, y):
    result = 1.0 / np.prod(y_predict.shape)
    result = result * self.derv_H_sigmoid(y_predict, y)    

    return result

  def __str__(self):
    return 'BINARY_CROSS_ENTROPY'

class ACCURACY:
  def predict(self, y_predict, y):
    est_yes = np.greater(y_predict, 0)
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