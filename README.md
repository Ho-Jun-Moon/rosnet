# rosnet

  

🇰🇷 ‘***rosnet***’ 은 ML을 적용한 causal discovery 패키지입니다. 제 개인 연구를 위해 만들었지만, 다른 사람들도 최대한 이해하기 쉽도록 설계했습니다. 

🔠 ‘***rosnet’*** is causal discovery package applied ML . I made it for my personal study. But, it is designed to be understanded as easy for others as possible. 
  
## 목적 / Purpose



**🇰🇷 이 패키지의 목적은 다음과 같습니다 :**

- ML 알고리즘을 Causal discovery에 적용
- 텐서 기반으로 기존 ML 알고리즘 재설계

**🔠 The purpose of this package is as follows :**

- Applying ML algorithm to Causal discovery
- Re-engineering existing ML algorithm based on tensor
  
## 설치 / Installment



```python
!pip install rosnet
```

**🔔 요구 패키지 / Required package**

- numpy
  
## 사용법 / Manual



🇰🇷 **이 패키지의 API는 *scikit-learn, keras* 와 거의 비슷합니다!** 

- 오직 `fit` 과 `predict`, 두 개의 함수만 사용하시면 됩니다.

🔠 **API of this package is just like *scikit-learn* and *keras*!**

- You only need to use two functions: `fit` and `predict`.

 
  
**예시 / Example** 

```python
# Multilayer Perceptron

# **Notice** : I made some ML algorithm as needed, but not all of them.
#          If you just want to use ML algorithm itself, 
#          it is recommened to use other ML packages like scikit-learn, tensorflow ...

from rosnet.neural_network import layers
import rosnet.neural_network as network

X_train = # Your code, numpy.narray expected 
y_train = # Your code, numpy.narray expected

def build_model():
  model = network.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1], )),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(4)
  ])

  optimizer = network.optimizers.SGD(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()
model.fit(X_train, y_train, 
					epochs=100, 
					batch_size = 1000, 
					validation_split = 0.2, 
					verbose = 0)
```
  
## 개발 기록 / Development log



### 0.0.1 - 22.03.26

- **rosnet.neural_network**
    - rosnet.neural_network.Sequential **add**
    - rosnet.neural_network.layers **add**
    - rosnet.neural_network.optimizers **add**
