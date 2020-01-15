import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# print(cancer.data.shape,cancer.target.shape)
# print(cancer.data[:3])
# plt.boxplot(cancer.data)
# plt.show()
# print(cancer.feature_names[[3,13,23]])

# print(np.unique(cancer.target, return_counts=True))

x = cancer.data
y = cancer.target

x_train, x_test , y_train, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

# print(x_train.shape,x_test.shape)

# print(np.unique(y_train,return_counts=True))

class LogisticNeuron:
    def __init__(self):
        self.w = None
        self.b = None
    
    def forpass(self,x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self,x,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        for i in range(epochs):
            for x_i, y_i in zip(x,y):
                z = self.forpass(x_i)
                a = self.activation(z)
                err = -(y_i-a)
                w_grad , b_grad = self.backprop(x_i,err)
                self.w -= w_grad
                self.b -= b_grad
    
    def activation(self,z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5

neuron = LogisticNeuron()
neuron.fit(x_train,y_train)

print(np.mean(neuron.predict(x_test) == y_test))