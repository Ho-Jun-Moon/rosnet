import numpy as np

from ..decorator import check_vector

class AdaStump:
  def __init__(self, classifier):
    self.classifier_ = classifier
  
  def fit(self, x, y, w = None):
    if len(y.shape) == 1:
      y = y.reshape(-1,1)
        
    # print(w)

    if w is None:
      #y가 너무 많으면 이거 0로 처리될 수도 있음
      self.w_input = np.ones(y.shape)/len(y)
      self.history_ = self.classifier_.fit(x, y)
      self.y_predict = self.classifier_.predict(x)
    else:
      # 여기서 뭘 해야 하냐면
      # 가중치까지 집어 넣어서 분류기의 파라미터를 피팅해야 한다.
      # 이거 모델 수정해서 w도 포함하도록 해야 겠는데
      if len(w.shape) == 1 :
        w = w.reshape(-1,1)
      self.w_input = w
      self.history_ = self.classifier_.fit(x, y, weights = w)
      self.y_predict = self.classifier_.predict(x)

      self.get_weight(self.y_predict, y)

      return self.w_output
  
  def predict(self, x):
    result = self.classifier_.predict(x)
    if len(result) == 1:
      result = result.reshape(-1,1)
    return result

  @check_vector
  def get_weight(self, y_predict, y):
    est_yes = np.greater(y_predict, 0.5)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.logical_and(est_yes, ans_yes)
    fp = np.logical_and(est_yes, ans_no)
    fn = np.logical_and(est_no, ans_yes)
    tn = np.logical_and(est_no, ans_no)

    correct = np.logical_or(tp, fn)
    incorrect = np.logical_or(tn, fp)

    self.tot_error_ = np.sum(incorrect * self.w_input)  
    self.amount_of_say_ = self.safe_div(1 - self.tot_error_, self.tot_error_)   
    self.amount_of_say_ = np.log(self.amount_of_say_) / 2 

    self.w_output = self.w_input
    self.w_output[incorrect] = self.w_output[incorrect] * np.exp(self.amount_of_say_)
    self.w_output[correct] = self.w_output[correct] * np.exp(-self.amount_of_say_)

    condition = np.isnan(self.w_output)
    self.w_output[condition] = 0

    self.w_output = self.w_output / np.sum(self.w_output)

  def safe_div(self, p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20: return np.sign(p)
    return p / q
  