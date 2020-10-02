from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

iris=datasets.load_iris()
n=0
x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int)

#creating classifer
clf=LogisticRegression()
#model fit is done hear for data x and y
clf.fit(x,y)

# prediction for the new dataset

demo=clf.predict(([[2.6]]))
print(demo)
x_new=np.linspace(0,3,1000).reshape(-1,1)
y_prob=clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1])
plt.show()
        