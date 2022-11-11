In[1]
from sklearn.datasets import load_breast_cancer

In[2]
cancer = load_breast_cancer()

In[3]
print(cancer.keys())

In[4]
cancer['target_names']

In[5]
cancer['feature_names']

In[6]
len(cancer['feature_names'])

In[7]
import numpy as np
import pandas as pd

In[8]
data= np.c_[cancer.data,cancer.target]
columns = np.append(cancer.feature_names,["target"])
Df=pd.DataFrame (data , columns= columns) 
print(Df.head())

In[9]
Df.tail()

In[10]
Df['target'].value_counts()
Df.head()

In[11]
x=Df[Df.columns[:-1]]
y=Df.target
x.head()

In[12]
y.tail(15)

In[13]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split (x,y, test_size=0.2, random_state=123, stratify=y)
x_train.shape

In[14]
x_test.shape

In[15]
# model training
from sklearn.svm import SVC
svm = SVC(C =1.0, gamma = 'auto', kernel = 'rbf',probability = True)
model= svm.fit(x_train, y_train)
y_predict= model.predict(x_test)
y_predict

In[16]
np.array(y_test)

In[17]
from sklearn import metrics
accuracy = metrics.accuracy_score(y_predict, y_test)
accuracy

In[18]
confusion = metrics.confusion_matrix(y_predict, y_test)
confusion

In[19]
from sklearn.metrics import classification_report, f1_score,recall_score,precision_score
precision_ = precision_score(y_test, y_predict)
print("precision:", precision_)
recall_= recall_score(y_test, y_predict)
print("recall:", recall_)

