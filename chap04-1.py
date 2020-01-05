#20200105 공부공부공부
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

class SingleLayer:
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []
    
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
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.forpass(x[i])
                a = self.activation(z)
                err = -(y[i]-a)
                w_grad , b_grad = self.backprop(x[i],err)
                self.w -= w_grad
                self.b -= b_grad
                a = np.clip(a,1e-10,1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            
    
    def activation(self,z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z))
        return a > 0.5
    
    def predict(self,x):
        z = [self.forpass(x_i) for x_i in x]
        return np.array(z) > 0
    
    def score(self,x,y):
        return np.mean(self.predict(x)==y)

layer = SingleLayer()
layer.fit(x_train,y_train)


# print(layer.score(x_test,y_test))

# plt.plot(layer.losses)
# plt.show()

sdg = SGDClassifier()