import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

x = diabetes.data[:,2]
y = diabetes.target

w = 1.0
b = 1.0

for i in range(1,100):
    for x_i, y_i in zip(x,y):
        y_hat = x_i * w + b
        err = y_i-y_hat
        w_rate = x_i
        w = w + w_rate*err
        b = b + 1 * err

plt.scatter(x,y)
pt1 = (-0.1,-0.1*w+b)
pt2 = (0.15,0.15*w+b)
plt.plot([pt1[0],pt2[0]],[pt1[1],pt2[1]])
plt.show()

