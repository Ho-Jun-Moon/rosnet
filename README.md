# rosnet
  
**rosnet** 은 ML(Machine Learning) & NN(Neural Network) 패키지입니다. 제 연구 영역인 causal discovery에서 자유자재로 ML & NN을 사용하기 위해 구현했습니다. 

필요하면 그때 그때 만들어서 구현되지 않은 ML & NN 기법들이 많습니다. 특별한 목적이 없으시다면, scikit learn이나 keras 같은 대형 패키지 사용을 추천드립니다.

### 특징

- **scikit-learn, keras 와 유사한 API 구조**
- **오직 Numpy 만으로 구현됨**
    - 직접 구현하면서 공부하려는 목적도 있어서 Numpy만 사용했습니다.

- **(계획) 텐서 중심의 알고리즘**
    - 이 부분은 아직 제 공부가 모자라 제대로 하지 못하고 있습니다.

### 설치

```python
!pip install rosnet
```

- 요구 패키지 : Numpy

### 사용법 및 구현 설명

시간이 날 때마다 explanation에 노트북 파일로 구현 설명을 적고 있습니다. 혹 ML 구현에 관심이 있으시다면, 도움이 될지도 모르겠습니다.

### 구현 Log

- ~ `0.3.2`
    - Basic NN 구현
    - Tree 구현
    - AdaBoost 구현
    - Adam optimizer 구현
    - 성능 측정에 필요한 Metrics 구현
