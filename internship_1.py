#!/usr/bin/env python
# coding: utf-8

# In[62]:


#importing imporatant libraries required for the task.
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[63]:


#using .read_csv to load the dataset into the code.
df = pd.read_csv('Downloads/Disease_symptom_and_patient_profile_dataset.csv')


# In[64]:


#checking if the dataset is properly loaded.
df.head()


# In[65]:


#MANIPULATING DATA
#checking the dataset for any missing values.
print(df.isnull().sum())


# In[66]:


#dropping any of the null values if present in the dataset.
df = df.dropna()


# In[67]:


#SPLITTING THE DATA INTO TRAINING AND TESTING SET
# as there are multiple features and one target variable the following lines code defines the columns in features and target_variable.
features = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
target_variable = 'Outcome Variable'

X = df[features]
y = df[target_variable]

# spliting the data into training and testing sets so it becomes easier to test the data.
# i have put the training set of data in 20 proportion, for better accuracy of the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#checking whether the data is properly split. 
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[68]:


#DECISION CLASSIFIER
# one-hot encode categorical features so that the data is all in correct and consistent format.
X = pd.get_dummies(df[features])


y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# now, decision tree classifier is applied to the data
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[69]:


# EVALUATION
y_pred = model.predict(X_test)

# finding out the accuracy, precision and recall of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[ ]:




