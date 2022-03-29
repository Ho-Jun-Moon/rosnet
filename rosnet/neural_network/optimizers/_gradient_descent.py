import numpy as np

class GD:
  def __init__(self, learning_rate = 0.001):
    self.learning_rate_ = learning_rate
  
  # 행렬 기반으로 작성함...
  def fit(self, x, y, layers, loss, metrics, metrics_per_batch):
    self.layers_ = layers
    self.loss_ = loss
    y_predict, Ys, Zs = self.forward(x)

    for key in metrics_per_batch.keys():
      metrics_per_batch[key].append(metrics[key].predict(y_predict, y))

    D = self.loss_.d(y_predict, y)
    for i in reversed(range(len(self.layers_))):
      D = D * self.layers_[i].activation_.d(Ys[i])
      dW = -self.learning_rate_ *  np.matmul(Zs[i].transpose(), D)
      dB = -self.learning_rate_ * np.sum(D, axis = 0)
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