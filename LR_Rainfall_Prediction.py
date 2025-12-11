
#importing libraries
import seaborn as sns
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import os
import time
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from sklearn import metrics
import seaborn as sns
from sklearn.svm import SVC


df = pd.read_csv('/content/weatherAUS.csv')
df.head(3)

# from google.colab import drive
# drive.mount('/content/drive')

print('Size of data: ',len(df))
#Export file to see data
# df.to_excel(r'Original_file.xlsx', index = False)

# Dropping rows with Null value in the target variable and RainToday variable
df.drop(df[pd.isnull(df['RainTomorrow'])].index, inplace=True)
df.drop(df[pd.isnull(df['RainToday'])].index, inplace=True)
df.head(3)

print(' Size of data (number of rows left) after dropping columns with null values in RainToday and RainTomorrow variables:\n Data size:',len(df))

# # Converting Categorical values to binary values in RainToday variable
tmp = []
for i in tqdm(range(len(df))):
  if (df.iloc[i]['RainToday']) == 'Yes':
    tmp.append(1)
  else:
    tmp.append(0)

rain_today = tmp
# print((rain_today))



# Converting Categorical values to binary values in Target/RainTomorrow variable
tmp = []
for i in tqdm(range(len(df))):
  if (df.iloc[i]['RainTomorrow']) == 'Yes':
    tmp.append(1)
  else:
    tmp.append(0)

target = tmp
# print((target))



#selecting variables or features for the model (select features that seems important/useful; like states/regions are dropped)
rain = df[['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm']].copy()
rain['RainToday'] = rain_today
print('Number of records: ',len(rain))
rain.head(3)



# Before imputation: Number of Null values in each columns
print('---------- Before imputation: Number of null values in each columns ----------')
print('\nTotal No. of records/rows: ', len(rain))
print('MinTemp:',pd.isnull(rain['MinTemp']).sum())
print('MaxTemp:',pd.isnull(rain['MaxTemp']).sum())
print('Rainfall:',pd.isnull(rain['Rainfall']).sum())
print('Evaporation:',pd.isnull(rain['Evaporation']).sum())
print('Sunshine:',pd.isnull(rain['Sunshine']).sum())
print('WindGustSpeed:',pd.isnull(rain['WindGustSpeed']).sum())
print('WindSpeed9am:',pd.isnull(rain['WindSpeed9am']).sum())
print('WindSpeed3pm:',pd.isnull(rain['WindSpeed3pm']).sum())
print('Humidity9am:',pd.isnull(rain['Humidity9am']).sum())
print('Humidity3pm:',pd.isnull(rain['Humidity3pm']).sum())
print('Pressure9am:',pd.isnull(rain['Pressure9am']).sum())
print('Pressure3pm:',pd.isnull(rain['Pressure3pm']).sum())
print('Cloud9am:',pd.isnull(rain['Cloud9am']).sum())
print('Cloud3pm:',pd.isnull(rain['Cloud3pm']).sum())
print('Temp9am:',pd.isnull(rain['Temp9am']).sum())
print('Temp3pm:',pd.isnull(rain['Temp3pm']).sum())

# Convert dataframe to numpy for imputation operation
rain_np = rain.to_numpy()
# print(rain_np)

# Training the imputation model [Multivariate feature imputation]
# Run time is around 2 minutes
st = time.time()
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=50, random_state=1)
imp.fit(rain_np)

end = time.time()
diff = end-st
print('time: ', round(diff,2),'sec')

# Transform Null values using Multivariate feature imputation
rain_np_imp  = imp.transform(rain_np)

# print(type(rain_np_imp))
# rain_np_imp[0]

# Convert numpy data back to dataframe data
rain_imp = pd.DataFrame(rain_np_imp, columns = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday'])
rain_imp.head(3)
# rain_imp.to_excel(r'Original_file_imp.xlsx', index = False)

# After imputation: Number of Null values in each columns should be zero
print('---------- After imputation: Number of null values in each columns should be zero ----------')
print('\nTotal No. of records/rows: ', len(rain_imp))
print('MinTemp:',pd.isnull(rain_imp['MinTemp']).sum())
print('MaxTemp:',pd.isnull(rain_imp['MaxTemp']).sum())
print('Rainfall:',pd.isnull(rain_imp['Rainfall']).sum())
print('Evaporation:',pd.isnull(rain_imp['Evaporation']).sum())
print('Sunshine:',pd.isnull(rain_imp['Sunshine']).sum())
print('WindGustSpeed:',pd.isnull(rain_imp['WindGustSpeed']).sum())
print('Humidity9am:',pd.isnull(rain_imp['Humidity9am']).sum())
print('Humidity3pm:',pd.isnull(rain_imp['Humidity3pm']).sum())
print('Pressure9am:',pd.isnull(rain_imp['Pressure9am']).sum())
print('Pressure3pm:',pd.isnull(rain_imp['Pressure3pm']).sum())
print('Cloud9am:',pd.isnull(rain_imp['Cloud9am']).sum())
print('Cloud3pm:',pd.isnull(rain_imp['Cloud3pm']).sum())
print('Temp9am:',pd.isnull(rain_imp['Temp9am']).sum())
print('Temp3pm:',pd.isnull(rain_imp['Temp3pm']).sum())

# (1). We are not using p-value for features selection because in backward method after 2nd iteration we reached a dead end
#      where all features have zero p-value.
#      I have documented the work(p-value) in doc file, please refer for detail (file name: Issues in data pre-processing.docx)
# (2). SelectBest method is also not effective because it just gives us training data with K features without any header,
#      difficult to identify features name.

#  For more details please see the document in the below link
# https://docs.google.com/document/d/15791_QsESwri4tQ2-tIYJV5tJWbKhvO5/edit?usp=sharing&ouid=113173886630120559872&rtpof=true&sd=true

# ------------ Random forest features selection ------------
# ---------------- Cell run time is 60 sec -----------------
from sklearn.ensemble import RandomForestClassifier

y = target
x = rain_imp
st = time.time()

forest = RandomForestClassifier(random_state=0)
forest.fit(x, y)

importances = forest.feature_importances_

end = time.time()
diff = end-st
print('time: ', round(diff,2),'sec')
print('Features selection')

# Header of the columns/features, used in graph plot
COLS = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday']


# Plot graph
forest_importances = pd.Series(importances, index=COLS)
fig, ax = plt.subplots()
forest_importances.plot.bar()
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

print('\nFeatures sorted as per importance:\n')
# Features importance sorted list
feat = []
for i in range(len(COLS)):
  feat.append((round(importances[i],3), COLS[i]))
feat.sort(reverse=True)

for i in feat:
  print(i[1],':',i[0],end=' || ')

print('\n\n')



# Feature selection: Six columns selected for the LR model based on MDI
rain_imp_feat = rain_imp[['Humidity3pm','Sunshine','Pressure3pm','Cloud3pm','Pressure9am','WindGustSpeed']].copy()
rain_imp_feat.head(3)

# Export file to see the data
# rain_imp_feat.to_excel(r'Feature_selected_imp.xlsx', index = False)

# Skew detection and removal

rain_imp_feat.hist(column=['Humidity3pm'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['Humidity3pm']).min(),3))
print('Max:',round((rain_imp_feat['Humidity3pm']).max(),3))

# After square transformation on variable 'Humidity3pm' our distribution changed from -ve skew to +ve

#  For more details please see the document in the below link
# https://docs.google.com/document/d/15791_QsESwri4tQ2-tIYJV5tJWbKhvO5/edit?usp=sharing&ouid=113173886630120559872&rtpof=true&sd=true

print('We are skipping this transformation because it flipped from -ve skew to +ve skew')

rain_imp_feat.hist(column=['Sunshine'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['Sunshine']).min(),3))
print('Max:',round((rain_imp_feat['Sunshine']).max(),3))

rain_imp_feat.hist(column=['Pressure3pm'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['Pressure3pm']).min(),3))
print('Max:',round((rain_imp_feat['Pressure3pm']).max(),3))

rain_imp_feat.hist(column=['Cloud3pm'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['Cloud3pm']).min(),3))
print('Max:',round((rain_imp_feat['Cloud3pm']).max(),3))

rain_imp_feat.hist(column=['Pressure9am'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['Pressure9am']).min(),3))
print('Max:',round((rain_imp_feat['Pressure9am']).max(),3))

rain_imp_feat.hist(column=['WindGustSpeed'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['WindGustSpeed']).min(),3))
print('Max:',round((rain_imp_feat['WindGustSpeed']).max(),3))

# # Log transformation to remove positive skewness
for i in tqdm(range(len(rain_imp_feat))):
  rain_imp_feat.iloc[i]['WindGustSpeed'] = math.log(rain_imp_feat.iloc[i]['WindGustSpeed'])

rain_imp_feat.hist(column=['WindGustSpeed'],bins=50, figsize=(7,5))
print('Min:',round((rain_imp_feat['WindGustSpeed']).min(),3))
print('Max:',round((rain_imp_feat['WindGustSpeed']).max(),3))



# Performing min-max normalization/scaling of data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
min_max_scaler = preprocessing.MinMaxScaler()
rain_norm = min_max_scaler.fit_transform(rain_imp_feat)

# Convert numpy to dataframe
rain_norm = pd.DataFrame(rain_norm, columns = ['Humidity3pm','Sunshine','Pressure3pm','Cloud3pm','Pressure9am','WindGustSpeed'])
rain_norm.head(3)

# Target variable(list) is converted to dataframe
df_tar = pd.DataFrame(target)
df_tar = df_tar.set_axis(['RainTomorrow'], axis=1)
# df_tar.head(3)

# Making a copy of data to be used in LR model
x_og = np.array(rain_norm.copy())
y_og = np.array(df_tar.copy())
y_og = y_og.reshape(140787,)
print(x_og.shape)
print(y_og.shape)

# ------------------------------------------- Logistic Regression Model Starts here -------------------------------------------
print('############## Logistic Regression Model Starts here ##############')

# Set the seed value
# change it to get random set of data
seed = 5

# split dataset into training test samples
x_train, x_val, y_train, y_val = train_test_split(x_og, y_og, test_size=(40/100), random_state = seed)

# Making an instance of the Model
lr_model = LogisticRegression(random_state = seed, solver='lbfgs',max_iter = 50000)

# training the model
lr_model.fit(x_train, y_train)

# Making prediction for training data
predictions_train = lr_model.predict(x_train)

# Making prediction for validation data
predictions = lr_model.predict(x_val)

# Calculating Accuracy for training data
accuracy_train = lr_model.score(x_train, y_train)
print('Accuracy of the LR model for training data is: ',(round((accuracy_train*100),2)))

# Calculating Accuracy validation data
accuracy = lr_model.score(x_val, y_val)
print('Accuracy of the LR model for validation data is: ',(round((accuracy*100),2)))

# Making confusion matrix
cm = metrics.confusion_matrix(y_val, predictions)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=1, square = True, cmap = 'Greens_r' );
plt.ylabel('Actual label',size=15);
plt.xlabel('Predicted label',size=15);
all_sample_title = 'Accuracy Score: {0}'.format(round(accuracy*100,2))
plt.title(all_sample_title, size = 15);



# ------------------------------------------- Logistic Regression Model from scratch without library function use -------------------------------------------
print('############## Logistic Regression Model from scratch without library function use ##############\n')

# Add bias/intercept column in LR equation
tmp = [1] * len(rain_norm)
rain_norm['Bias'] = tmp

# Making a copy of data to be used in LR model
x_ogs = np.array(rain_norm.copy())
y_ogs = np.array(df_tar.copy())
y_ogs = y_ogs.reshape(len(y_ogs),)
print(x_ogs.shape)
print(y_ogs.shape)

# split dataset into training test samples: training - 70%; validation - 30%
x_train, x_val, y_train, y_val = train_test_split(x_ogs, y_ogs, test_size=(30/100), random_state = seed)
print('Data size:',len(x_ogs))
print('Train data size:',len(x_train))
print('Validation data size:',len(x_val))

# initializing weight/beta coeff array
tmp = [0] * (7)
wt = np.array([tmp])
print('Weight array:',wt)

# Set number of iterations
iteration = 9000
print('Iterations: ',iteration)

# Set learning rate
lr = 0.03
print('Learning rate: ',lr)

#hypothesis/prediction function
def prediction(feature, wt):
    feature = feature.transpose()
    z = np.dot(wt, feature)
    return 1.0 / (1 + np.exp(-z))


#Cost function
def cost(pred, real):
    c = (np.dot(np.log(pred), real) + np.dot(np.log(1-pred), (1-real)))/(-len(pred))
    return (c)


#gradient descent function
def grad(w, lr, real, pred, feat):
    global wt
    real = real.transpose()
    gradient = (np.dot((pred - real),feat))/len(real)

    #adjusting weight
    wt = w - (lr * gradient)

#cost array
cst1 = []
cst2 = []

#accuracy array
act = []  #train
acv = []  #val

#accuracy function
def accuracy(features, real):
    tt = prediction(features, wt)
    acc = ((np.where(tt >= 0.5, 1, 0) == real).sum())/len(real)
    return acc

# Training the model
# Takes around 3-5 minutes for 10k iterations

st = time.time()

for i in tqdm(range(iteration)):
    h_train = prediction(x_train, wt)
    h_val = prediction(x_val, wt)
    grad(wt, lr, y_train, h_train, x_train)
    cst1.append(cost(h_train, y_train))
    cst2.append(cost(h_val, y_val))
    act.append(accuracy(x_train, y_train))
    acv.append(accuracy(x_val, y_val))

end = time.time()
diff = end-st
print('\ntime: ', round(diff,2),'sec')

#calling accuracy function to get accuracy
train_acc = accuracy(x_train, y_train)
train_acc = round((train_acc * 100),2)
print('\nTraining data accuracy: ',train_acc)

val_acc = accuracy(x_val, y_val)
val_acc = round((val_acc * 100),2)
print('\nValidation data accuracy: ',val_acc)

#cost function plot
#print('\nCost function progress:\n')
plt.figure(figsize=(6, 4))
plt.plot(np.arange(iteration), cst1 , label ='training', linewidth = 2)
plt.plot(np.arange(iteration), cst2, label = 'validation',  linewidth = 2)
plt.xlabel('Iteration', size=20)
plt.ylabel('Cost', size=20)
plt.legend(loc='upper right')
plt.grid()
plt.show(block=False)
plt.pause(7)
plt.close()

#Accuracy function plot
print('\n')
plt.figure(figsize=(6, 4))
plt.plot(np.arange(iteration), act , label ='training', linewidth = 2)
plt.plot(np.arange(iteration), acv, label = 'validation', linewidth = 2)
plt.xlabel('Iteration', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(loc='lower right')
plt.grid()
plt.show(block=False)
plt.pause(7)
plt.close()


print('\nTraining data accuracy: ',train_acc)
print('\nValidation data accuracy: ',val_acc)

# Making confusion matrix
predictions = np.where(prediction(x_val, wt) >= 0.5, 1, 0)
cm = metrics.confusion_matrix(y_val, predictions[0])

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=1, square = True, cmap = 'Greens_r' );
plt.ylabel('Actual label',size=15);
plt.xlabel('Predicted label',size=15);
all_sample_title = 'Accuracy Score: {0}'.format(val_acc)
plt.title(all_sample_title, size = 15);

# Display beta coefficient or weight of features
weight = ['Humidity3pm', 'Sunshine', 'Pressure3pm', 'Cloud3pm', 'Pressure9am', 'WindGustSpeed']
for i in range(len(weight)):
  print(weight[i],':',round(wt[0][i],2))



