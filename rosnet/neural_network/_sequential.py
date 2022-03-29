from .optimizers import get_optimizer
from ..metrics import get_metrics

import numpy as np

class Sequential:
  def __init__(self, layers =None):
    if layers != None:
      self.layers = layers
      self.set_input_count_all()
    else: 
      self.layers = []
  
  def add(self, layer):
    if len(self.layers) == 0:
      self.layers.append(layer)

    else:
      output = self.layers[-1].output_shape_
      layer.set_input(output)
      self.layers.append(layer)

  def set_input_count_all(self):
    for i in range(1, len(self.layers)):
      output = self.layers[i-1].output_shape_
      self.layers[i].set_input(output)

  def compile(self, loss, optimizer, metrics, batch_size=None, epochs=1,verbose=1, validation_split=0.0, early_stop = None, is_shuffle = True):
    
    self.metrics_ = {}
    self.is_shuffle_ = is_shuffle

    if type(loss) == str:
      self.metrics_['loss'] = get_metrics(loss)()
      self.loss_ = get_metrics(loss)()

    else:
      self.loss_ = loss()
      self.metrics_['loss'] = loss()
    
    if type(optimizer) == str:
      self.optimizer_ = get_optimizer(optimizer)()
    else:
      self.optimizer_ = optimizer
    
    
    for key in metrics:
      if type(key) == str:
        self.metrics_[key] = get_metrics(key)()
      else:
        tmp = key()
        self.metrics_[str(tmp)] = tmp

    self.history_ = {}

    self.batch_size_ = batch_size
    self.epochs_ = epochs
    self.verbose_ = verbose
    self.validation_split_ = validation_split
    self.early_stop_ = early_stop
    
  
  def fit(self, x, y, batch_size=None, epochs=None,verbose=None, validation_split=None, early_stop = None, weights = None, is_shuffle = None):
    
    if len(y.shape) == 1:
      y = y.reshape(-1,1)

    if batch_size is None:
      batch_size = self.batch_size_
    if epochs is None:
      epochs = self.epochs_
    if verbose is None:
      verbose = self.verbose_
    if validation_split is None:
      validation_split = self.validation_split_
    if early_stop is None:
      early_stop = self.early_stop_
    if weights is not None:
      self.loss_.set_weight(weights)
    if is_shuffle is None:
      is_shuffle = self.is_shuffle_

    self.metrics_per_batch = {}

    for key in self.metrics_.keys():
      if validation_split != 0.0: 
        self.history_['val_' + key] = []
      self.history_[key] = []
      self.metrics_per_batch[key] = []
      
       
    unit = int(x.shape[0] * validation_split)
    shuffle = np.arange(x.shape[0])
    if is_shuffle:
      np.random.shuffle(shuffle)
    x_val = x[shuffle[:unit]]
    y_val = y[shuffle[:unit]]
    x_train = x[shuffle[unit:]]
    y_train = y[shuffle[unit:]]

    # epoch
    if batch_size is None:
      batch_size = x_train.shape[0]
      steps = 1
    else:
      steps = int(x_train.shape[0]//batch_size)

    for epoch in range(epochs):

      # 데이터 저장용으로 dictionary 하나 정의
      for key in self.metrics_.keys():
        self.metrics_per_batch[key] = []

      # 매 epoch 마다 데이터를 섞어서 진행한다.
      shuffle = np.arange(x_train.shape[0])
      for step in range(steps):
        if step == 0 and is_shuffle: np.random.shuffle(shuffle)
        x_train_train = x_train[shuffle[batch_size*step : batch_size * (step + 1)]]
        y_train_train = y_train[shuffle[batch_size*step : batch_size * (step + 1)]]
        
        self.fit_(x_train_train, y_train_train)

      self.write_history(x_val, y_val)

      if verbose > 0 and (epoch + 1)%verbose ==0:
        result = f"[Epoch {epoch}] " + str([f"{key} = {self.history_[key][-1]:.3f}" for key in self.history_.keys()])
        print(result)     
    if verbose > 0:
      print("\n[Final]" + str([f"{key} = {self.history_[key][-1]:.3f}" for key in self.history_.keys()]))

  def fit_(self, x, y):
    self.layers, self.metrics_per_batch = self.optimizer_.fit(x, y, layers = self.layers, loss = self.loss_, metrics = self.metrics_, metrics_per_batch = self.metrics_per_batch)

  def write_history(self, x_val, y_val):
    if len(x_val) != 0 :
      y_predict = self.predict(x_val)
      for key in self.history_.keys():
        if key[:3] == 'val':
          met = self.metrics_[key[4:]].predict(y_predict, y_val)
          self.history_[key].append(met)
        else:
          met = np.mean(self.metrics_per_batch[key])
          self.history_[key].append(met)
    else:
      for key in self.history_.keys():
        if key[:3] == 'val':
          continue
        else:
          met = np.mean(self.metrics_per_batch[key])
          self.history_[key].append(met)

    self.metrics_per_batch = {}

  def predict(self, x):
    Z = x
    for layer in self.layers:
      Z = layer.predict(Z)

    return Z