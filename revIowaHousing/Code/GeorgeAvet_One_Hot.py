import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")

data.head()

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define Condition 1 
data_Condition1 = data['Condition1']
# values = array(data)
# print(values)
# # integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data_Condition1)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_Condition1 = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded_Condition1)



# define Condition 2 
data_Condition2 = data['Condition2']
# values = array(data)
# print(values)
# # integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data_Condition1)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_Condition2 = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded_Condition2)



