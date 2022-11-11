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
