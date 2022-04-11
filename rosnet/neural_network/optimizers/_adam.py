import numpy as np

class ADAM:
  def __init__(self, learning_rate = 0.001, b1 = 0.9, b2 = 0.999, epsilon = 1.0e-8):
    self.b1_ = b1
    self.b2_ = b2
    self.epsilon_ = epsilon
    self.learning_rate_ = learning_rate
    self.step = 0
    self.first = True
    
  
  # 행렬 기반으로 작성함...
  def fit(self, x, y, layers, loss, metrics, metrics_per_batch):
    self.layers_ = layers
    self.loss_ = loss
    if self.first:
        self.first = False
        self.mW = [np.zeros(layer.W.shape) for layer in self.layers_]
        self.mB = [np.zeros(layer.B.shape) for layer in self.layers_]
        self.vW = [np.zeros(layer.W.shape) for layer in self.layers_]
        self.vB = [np.zeros(layer.B.shape) for layer in self.layers_]

    self.step += 1   

    y_predict, Ys, Zs = self.forward(x)

    for key in metrics_per_batch.keys():
      metrics_per_batch[key].append(metrics[key].predict(y_predict, y))

    D = self.loss_.d(y_predict, y)
    for i in reversed(range(len(self.layers_))):
      D = D * self.layers_[i].activation_.d(Ys[i])

      gW = np.matmul(Zs[i].transpose(), D)
      gB = np.sum(D, axis = 0)

      self.mW[i] = self.mW[i] * self.b1_ + (1-self.b1_) * gW
      self.vW[i] = self.vW[i] * self.b1_ + (1-self.b2_) * (gW * gW)
      self.mB[i] = self.mB[i] * self.b1_ + (1-self.b1_) * gB
      self.vB[i] = self.vB[i] * self.b1_ + (1-self.b2_) * (gB * gB)
      
      momentW = self.mW[i] / (1- np.power(self.b1_, self.step))
      second_momentW = self.vW[i] / (1- np.power(self.b2_, self.step))
      momentB = self.mB[i] / (1- np.power(self.b1_, self.step))
      second_momentB = self.vB[i] / (1- np.power(self.b2_, self.step))
      
      dW = -self.learning_rate_ * momentW / (np.sqrt(second_momentW) + self.epsilon_)
      dB = -self.learning_rate_ * momentB / (np.sqrt(second_momentB) + self.epsilon_)

    #   dW = -self.learning_rate_ *  np.matmul(Zs[i].transpose(), D)
    #   dB = -self.learning_rate_ * np.sum(D, axis = 0)

      D = np.matmul(D, self.layers_[i].W.transpose())

      self.layers_[i].back(dW, dB) #결과를 저장해야 한다.

    return self.layers_, metrics_per_batch

  def forward(self, x):
    Zs = []
    Ys = []
    Z_old = x
    for layer in self.layers_:
      Zs.append(Z_old)
      Y, Z_new = layer.forward(Z_old)
      Ys.append(Y)
      Z_old = Z_new

    # Zs.append(Z_new) # 마지막 출력을 쓴다.
    if len(Z_new) == 1:
      Z_new = Z_new.reshape(-1,1)

    return Z_new, Ys, Zs
  
  def __str__(self):
    return "GD"