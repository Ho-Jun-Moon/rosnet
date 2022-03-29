import numpy as np

from ._adastump import AdaStump
from ..metrics import get_metrics
from ..util import train_test_split

class AdaBoost:
  def __init__(self, classifiers):
    self.classifiers_ = classifiers
    self.estimators_ = []
  
  def compile(self, loss, metrics, n_estimators = 50, verbose=1, validation_split=0.0, early_stop = None, is_shuffle = False):
    
    self.metrics_ = {}

    if type(loss) == str:
      self.metrics_['loss'] = get_metrics(loss)()
      self.loss_ = get_metrics(loss)()

    else:
      self.loss_ = loss()
      self.metrics_['loss'] = loss()
    
    for key in metrics:
      if type(key) == str:
        self.metrics_[key] = get_metrics(key)()
      else:
        tmp = key()
        self.metrics_[str(tmp)] = tmp

    self.history_ = {}

    self.verbose_ = verbose
    self.validation_split_ = validation_split
    self.early_stop_ = early_stop
    self.is_shuffle_ = is_shuffle
    self.n_estimators_ = n_estimators
  
  def predict(self, x):
    result = np.zeros(x.shape[0]).reshape(-1,1)
    for estimator in self.estimators_:
      y_predict = estimator.predict(x)

      condition = y_predict > 0.5
      y_predict[condition] = 1 * estimator.amount_of_say_
      y_predict[~condition] = -1 * estimator.amount_of_say_

      result = result + y_predict
    
    result = np.greater(result, 0).astype(int)

    return result

  def fit(self, x, y, n_estimators = None, verbose=None, validation_split=None, early_stop = None, is_shuffle = None):
    
    if len(y.shape) == 1:
      y = y.reshape(-1,1)

    if verbose is None:
      verbose = self.verbose_
    if validation_split is None:
      validation_split = self.validation_split_
    if early_stop is None:
      early_stop = self.early_stop_
    if is_shuffle is None:
      is_shuffle = self.is_shuffle_
    if n_estimators is None:
      n_estimators = self.n_estimators_
    
    x_train, x_val, y_train, y_val = train_test_split(x, y, validation_split, is_shuffle)
    
    self.make_history()

    step = 0
    n_classifiers = len(self.classifiers_)
    #y가 너무 많으면 이거 0로 처리될 수도 있음
    w = np.ones(y.shape)/len(y)
    while step < n_estimators:
      self.make_metrics_per_step()

      stump = AdaStump(self.classifiers_[step % n_classifiers])
      self.estimators_.append(stump)
      w = self.estimators_[-1].fit(x, y, w)

      y_predict = self.predict(x)

      self.write_metrics_per_step(y_predict, y)
      self.write_history(x_val, y_val)
      
      if verbose > 0 and (step + 1)%verbose ==0:
        result = f"[Stump {step}] " + str([f"{key} = {self.history_[key][-1]:.3f}" for key in self.history_.keys()])
        print(result)     

      if self.history_['loss'][-1] == 0:
        break
      step += 1
    if verbose > 0 :
      print("\n[Final]" + str([f"{key} = {self.history_[key][-1]:.3f}" for key in self.history_.keys()]))

  def make_history(self):
    for key in self.metrics_.keys():
      if self.validation_split_ != 0.0: 
        self.history_['val_' + key] = []
      self.history_[key] = []

  def make_metrics_per_step(self):
    self.metrics_per_step = {}
    for key in self.metrics_.keys():
        self.metrics_per_step[key] = []

  def write_history(self, x_val, y_val):
      if len(y_val) != 0:
        y_predict = self.predict(x_val)
        for key in self.history_.keys():
          if key[:3] == 'val':
            met = self.metrics_[key[4:]].predict(y_predict, y_val)
            self.history_[key].append(met)
          else:
            met = self.metrics_per_step[key][-1]
            self.history_[key].append(met)
      else:
        for key in self.history_.keys():
          if key[:3] == 'val':
            continue
          else:
            met = self.metrics_per_step[key][-1]
            self.history_[key].append(met)
  
  def write_metrics_per_step(self, y_predict, y):
    for key in self.metrics_per_step.keys():
      met = self.metrics_[key].predict(y_predict, y)
      self.metrics_per_step[key].append(met)

  

  