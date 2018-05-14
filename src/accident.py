import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('accidents2016.csv', low_memory=False)


relevant_features = [
    "Weather_Conditions",
    "Light_Conditions",
    "Day_of_Week",
    "Road_Type",
    "Road_Surface_Conditions",
    "Urban_or_Rural_Area"
]
data = data[relevant_features]
data = pd.get_dummies(data, columns=['Weather_Conditions', 'Light_Conditions', 'Day_of_Week', 'Road_Type',
                                     'Road_Surface_Conditions', 'Urban_or_Rural_Area'])
data['label'] = 1


target = data['label']
outliers = data[target == -1]
print("outliers.shape", outliers.shape)
print("outlier fraction", outliers.shape[0] / target.shape[0])


target = data['label']


data.drop(["label"], axis=1, inplace=True)

train_data, test_data, train_target, test_target = train_test_split(data, target, train_size=0.8)
train_data.shape


from sklearn import svm

# set nu (which should be the proportion of outliers in our dataset)
nu = outliers.shape[0] / target.shape[0]
print("nu", nu)


model = svm.OneClassSVM(nu=0.0001, kernel='rbf', gamma=0.00005)
model.fit(train_data)


from sklearn import metrics


#training data
preds = model.predict(train_data)
targs = train_target

print("--------------Performance with training data----------------")
print("accuracy: ", metrics.accuracy_score(targs, preds))
print("precision: ", metrics.precision_score(targs, preds))
print("recall: ", metrics.recall_score(targs, preds))
print("f1: ", metrics.f1_score(targs, preds))


#test data
preds = model.predict(test_data)
targs = test_target

print("--------------Performance with test data----------------")
print("accuracy: ", metrics.accuracy_score(targs, preds))
print("precision: ", metrics.precision_score(targs, preds))
print("recall: ", metrics.recall_score(targs, preds))
print("f1: ", metrics.f1_score(targs, preds))


