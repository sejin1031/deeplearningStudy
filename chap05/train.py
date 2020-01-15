from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train_all, x_test, y_train_all, y_test = train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)

from sklearn.linear_model import SGDClassifier
sdg = SGDClassifier(loss ='log',random_state=42)
sdg.fit(x_train_all,y_train_all)
print(sdg.score(x_test,y_test))