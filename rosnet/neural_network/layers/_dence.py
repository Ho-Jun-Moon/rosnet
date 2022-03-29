from ..activation import get_activation

import numpy as np

class Dence:
  def __init__(self, output_shape, activation = 'linear', input_shape = None):
    self.output_shape_ = output_shape

    if type(activation) == str:
      self.activation_ = get_activation(activation)()
    else:
      self.activation_ = activation()
    
    if input_shape is not None:
      self.input_shape_ = input_shape[0]
      self.W = np.random.normal(0, 0.003, [self.input_shape_, self.output_shape_])
      self.B = np.zeros([self.output_shape_])

  def set_input(self, input):
    self.input_shape_ = input
    self.W = np.random.normal(0, 0.003, [self.input_shape_, self.output_shape_])
    self.B = np.zeros([self.output_shape_])

  # 이거 행렬 기반으로 작성해서 텐서로 바꾸려면 이거 하나 바꾸면 된다.
  def forward(self, x):
    Y = np.matmul(x, self.W) + self.B
    return Y, self.activation_.predict(Y)
  
  def predict(self, x):
    Y = np.matmul(x, self.W) + self.B
    return self.activation_.predict(Y)

  def back(self, dW, dB):
    self.W = self.W + dW
    self.B = self.B + dB