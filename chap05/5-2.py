#20200107 공부공부공부
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()




x = cancer.data
y = cancer.target

x_train, x_val , y_train, y_val = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)


class SingleLayer:
    def __init__(self,learning_rate=0.1):
        self.w = None
        self.b = None
        self.losses = []
        self.val_losses = []
        self.w_history = []
        self.lr = learning_rate
    
    def forpass(self,x):
        z = np.sum(x*self.w) + self.b
        return z
    
    def backprop(self,x,err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def fit(self, x, y, epochs=100, x_val = None, y_val = None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(self.w.copy())
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
                self.w_history.append(self.w.copy())
                a = np.clip(a,1e-10,1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            self.losses.append(loss/len(y))
            self.update_val_loss(x_val,y_val)
            
    def update_val_loss(self,x_val,y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.forpass(x_val[i])
            a = self.activation(z)
            a = np.clip(a,1e-10,1-1e-10)
            val_loss += -(y_val[i]*np.log(a) + (1-y_val[i])*np.log(1-a))
        self.val_losses.append(val_loss/len(y_val[]))

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



train_mean = np.mean(x_train,axis=0)
train_std = np.std(x_train,axis=0)
x_train_scaled = (x_train - train_mean) / train_std


layer2 = SingleLayer()
layer2.fit(x_train_scaled,y_train)
w2 = []
w3 = []

for w in layer2.w_history:
    w2.append(w[2])
    w3.append(w[3])
# plt.plot(w2,w3)
# plt.plot(w2[-1],w3[-1],'ro')


val_mean = np.mean(x_val,axis=0)
val_std = np.std(x_val,axis=0)
x_val_scaled = (x_val-val_mean)/val_std
print(layer2.score(x_val_scaled,y_val))

# plt.show()