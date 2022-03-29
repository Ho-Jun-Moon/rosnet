import numpy as np

def check_vector(function):
      def check(self, y_predict, y):
        if len(y_predict.shape) == 1:
            y_predict = y_predict.reshape(-1,1)
        if len(y.shape) == 1:
            y = y.reshape(-1,1) 
        result = function(self, y_predict, y)
        return result
      return check