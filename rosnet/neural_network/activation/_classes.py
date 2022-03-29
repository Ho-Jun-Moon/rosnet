import numpy as np

class RELU:
  def predict(self, x): 
    result = np.maximum(x,0)
    return result
  
  def d(self, Y):
    result = (Y > 0).astype(int)
    return result
  
  def __str__(self):
    return 'RELU'

class SIGMOID:
  def predict(self, x): 
    result = np.exp(-self.relu(-x))/(1.0 + np.exp(-np.abs(x)))
    return result
  
  def relu(self, X):
    return np.maximum(X,0)

  def d(self, Y):
    result = self.predict(Y)*(1-self.predict(Y))
    return result

    # result = np.ones(Y.shape)
    # return result

  def __str__(self):
    return 'SIGMOID'

class LINEAR:
  def predict(self, x): 
    result = x
    return x
  
  def d(self, Y):
    result = np.ones(Y.shape)
    return result

  def __str__(self):
    return 'LINEAR'