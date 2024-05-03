# DL-Assignment-1-Fashion-MNIST-Classification

!pip install visualkeras

 !pip install keras.utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('/content/drive/MyDrive/fashion-mnist_train.csv')
test = pd.read_csv('/content/drive/MyDrive/fashion-mnist_test.csv')

train.head()

test.head()

print(f"rows: {train.shape[0]}\ncolumns: {train.shape[1]}")
print(f"missing values: {train.isnull().any().sum()}")
train.head()

print(f"rows: {test.shape[0]}\ncolumns: {test.shape[1]}")
print(f"missing values: {test.isnull().any().sum()}")
test.head()

classes = {0 : 'T-shirt/top',
1 :  'Trouser',
2 : 'Pullover',
3 : 'Dress',
4 : 'Coat',
5 : 'Sandal',
6 : 'Shirt',
7 : 'Sneaker',
8 : 'Bag',
9 : 'Ankle boot'}

train['label_names'] = train['label'].map(classes)

plt.figure(figsize=(17,6))
sns.countplot(x = 'label_names', data = train)
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Training set images wrt classes')

test['label_names'] = test['label'].map(classes)

plt.figure(figsize=(17,6))
sns.countplot(x = 'label_names', data = test)
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Test set images wrt classes')

train = train.fillna(0)
test = test.fillna(0)

# Data cleaning #

print(train.isnull().sum().sum())
print(test.isnull().sum().sum())

train

test

Let's prepare our data for preprocessing. First, we're going to divide training and test sets into features and labels.

X_train = train.drop(['label', 'label_names'], axis = 1)
y_train = train.label

X_test = test.drop(['label', 'label_names'], axis = 1)
y_test = test.label
Now we're going to reshape our images

X_train = X_train.values.reshape(X_train.shape[0], 28, 28)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28)

X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)

print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
X_train shape:  (2839, 28, 28)
X_test shape:  (1886, 28, 28)
y_train shape:  (2839, 10)
y_test shape:  (1886, 10)

plt.figure(figsize=(20,5))
for i in range(5):
    val=''
    for j in range(10):
        if y_train[i][j]==1:
            val = classes[j]

    plt.subplot(int(f"15{i+1}")), plt.imshow(X_train[i], cmap='gray'), plt.axis('off'), plt.title(val, size=14)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(28, 28, 1), padding = 'same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding = 'same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

visualkeras.layered_view(model)

# Optimizer specified here is adam, loss is categorical crossentrophy and metric is accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=256, epochs=10, validation_data=(X_test, y_test))

fig, ax = plt.subplots(figsize=(15,5))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()

score = model.evaluate(X_test, y_test, steps=math.ceil(10000/32))
# checking the test loss and test accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score1 = model.evaluate(X_train, y_train, steps=math.ceil(10000/32))
# checking the test loss and test accuracy
print('Train loss:', score1[0])
print('Train accuracy:', score1[1])

from sklearn.metrics import confusion_matrix, classification_report

predict=model.predict(X_test)

y_pred=[]
for i in range(len(predict)):
    y_pred.append(np.argmax(predict[i]))

y = np.argmax(y_test, axis=-1)

cr = classification_report(y, y_pred)
print(cr)

 from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
 print(f"Accuracy: {accuracy_score(y, y_pred)}")
 print(f"Recall: {recall_score(y, y_pred, average='weighted')}")
 print(f"Precision: {precision_score(y, y_pred, average='weighted')}")
 print(f"F1 score: {f1_score(y, y_pred, average='weighted')}")

cm=confusion_matrix(y, y_pred)

f, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.01, linecolor='grey')
plt.title('Confustion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

fig, axis = plt.subplots(5, 5, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    pred_val=predict[i].argmax()

    ax.imshow(X_test[i], cmap='gray'), ax.axis('off')
    ax.set(title = f"Real Class: {classes[y_test[i].argmax()]}\nPredict Class: {classes[pred_val]}")

predicted_label = np.argmax(predict[0])

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.imshow(X_test[0], cmap='gray')
plt.xlabel(f"{classes[predicted_label]} - {100*predict[0][predicted_label]:2.0f}% - ({classes[y_test[0].argmax()]})", fontsize=18)
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
bar=plt.bar(list(classes.values()), predict[0], color="#777777")
bar[predicted_label].set_color('blue')
plt.yticks([])
plt.grid(False)

num_rows = 10
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    predicted_label = np.argmax(predict[i])
    true_label = np.argmax(y_test[i])

    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(X_test[i], cmap='gray')
    plt.xlabel(f"{classes[predicted_label]} - {100*predict[i][predicted_label]:2.0f}% - ({classes[true_label]})", fontsize=12)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    bar=plt.bar(list(classes.values()), predict[i], color="#777777")
    bar[true_label].set_color('red')
    bar[predicted_label].set_color('blue')
    plt.axis('off')

print("\n Accuracy : ",score[1]*100)

Hyper Parameter Tuning with Keras Tuner and reducing the overfitting problem with batch normalization and adding dropout Layers

pip install keras-tuner --upgrade

import keras_tuner
import keras

def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1, activation='relu'))
  model.compile(loss='mse')
  return model

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras.layers import LeakyReLU

from kerastuner import RandomSearch
tensorflow.keras.optimizers

from sklearn.kernel_ridge import KernelRidge
from keras import layers
from keras import regularizers
from keras.regularizers import l1, l2, l1_l2
from  tensorflow.keras.optimizers.legacy import Adam
# Learning rate, overfitting early stopping and plotting the accu. #
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

def build_model(hp):
    model_2 = keras.Sequential()
    model_2.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,4,5]),padding='same',
        input_shape=(28,28,1))
    )
    model_2.add(BatchNormalization())
    model_2.add(Activation(LeakyReLU(alpha=0.2)))
    model_2.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,4,5]), padding='same')
    )
    model_2.add(BatchNormalization())
    model_2.add(Activation(LeakyReLU(alpha=0.2)))
    model_2.add(Conv2D(
        filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_3_kernel', values = [3,4,5]),padding='same')
    )
    model_2.add(BatchNormalization())
    model_2.add(Activation(LeakyReLU(alpha=0.2)))
    model_2.add(MaxPooling2D(pool_size=(2, 2))
    )
    model_2.add(Flatten()
    )
    model_2.add(Dense(
                units=hp.Int('dense_1_units', min_value=32, max_value=320, step=16),
        activation=LeakyReLU(alpha=0.2))
    )
    model_2.add(Dropout(0.50))
    model_2.add(Dense(
        units=hp.Int('dense_2_units', min_value=32, max_value=240, step=16),
        activation=LeakyReLU(alpha=0.2))
    )
    model_2.add(Dropout(0.50))
    model_2.add(Dense(
        units=hp.Int('dense_3_units', min_value=32, max_value=160, step=16),
        activation=LeakyReLU(alpha=0.2))
    )
    model_2.add(Dropout(0.50))
    model_2.add(Dense(10,activation='softmax')
    )
    model_2.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy']
    )

    return model_2

# Building a checkpoint, which will try to save the model which performed the best on the validation dataset
check_point = ModelCheckpoint("best_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True)

from keras.layers import Activation, Dense
from keras.layers import MaxPooling2D
tuner_search=RandomSearch(build_model,
                          objective='val_accuracy',
                          max_trials=3,directory='output',project_name="Mnist_Fashion")

tuner_search.search(X_train,y_train,epochs=20,validation_data=(X_test,y_test),callbacks=[check_point])

# Based on the best validation accuracy, we are taking that model for prediction on X_test #

model4=tuner_search.get_best_models(num_models=1)[0]
model4.summary()

model4.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20)

score = model4.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score1 = model4.evaluate(X_train, y_train)

CNN 84.94% TEST ACCURACCY

CNN 95.24% TRAIN ACCURACCY

ML

MODEL

Train = pd.read_csv("/content/drive/MyDrive/fashion-mnist_train.csv") # main train data
dataTrain = Train.copy() # to protect main data, it is same as main data

Test = pd.read_csv("/content/drive/MyDrive/fashion-mnist_test.csv") # main test data
Test = Test.drop("label",axis=1)
dataTest = Test.copy() # to protect main data, it is same as main data
TRAIN

print(dataTrain.shape)

print(dataTrain.columns)

print(dataTrain.columns.value_counts().sum())

print(dataTrain.info())

print(dataTrain.index)

print(type(dataTrain))

print(dataTrain.isnull().sum().sum())

# Drop the rows that have `NaN` values
dataTrain.dropna(inplace=True)

print(dataTrain.duplicated().sum())

# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

TEST

print(dataTest.shape)

print(dataTest.columns)

print(dataTest.info())

print(dataTest.index)
RangeIndex(start=0, stop=1886, step=1)

print(type(dataTest))

print(dataTest.isnull().sum().sum())

# Drop the rows that have `NaN` values
dataTest.dropna(inplace=True)

print(dataTest.isnull().sum().sum())

print(dataTest.duplicated().sum())

# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

TRAIN AND TEST FOR TRAINING

DEFINITION

x = dataTrain.drop("label",axis=1)
y = dataTrain["label"]
SEPARATION

xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.2,random_state=42)
CONTROLLING

print(xTrain.shape)

print(xTest.shape)

print(yTrain.shape)

print(yTest.shape)

STANDARDIZATION

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

xTrain = scaler.fit_transform(xTrain)
xTest = scaler.fit_transform(xTest)

xTrain = xTrain.astype("float32") / 255

print(xTrain.shape) # checking shape

print(xTrain.ndim)

print(xTrain.dtype)

xTest = xTest.astype("float32") / 255

print(xTest.shape) # checking shape

print(xTest.ndim) # checking dimensions

print(xTest.dtype) # checking type

figure = plt.figure(figsize=(15,8))
digitTrain1 = xTrain[4]
digitTrain1 = np.array(digitTrain1)
digitTrain1 = digitTrain1.reshape((28,28))
plt.imshow(digitTrain1,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTrain2 = xTrain[10]
digitTrain2 = np.array(digitTrain2)
digitTrain2 = digitTrain2.reshape((28,28))
plt.imshow(digitTrain2,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTrain3 = xTrain[100]
digitTrain3 = np.array(digitTrain3)
digitTrain3 = digitTrain3.reshape((28,28))
plt.imshow(digitTrain3,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTrain4 = xTrain[400]
digitTrain4 = np.array(digitTrain4)
digitTrain4 = digitTrain4.reshape((28,28))
plt.imshow(digitTrain4,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTest1 = xTest[4]
digitTest1 = np.array(digitTest1)
digitTest1 = digitTest1.reshape((28,28))
plt.imshow(digitTest1,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTest2 = xTest[40]
digitTest2 = np.array(digitTest1)
digitTest2 = digitTest2.reshape((28,28))
plt.imshow(digitTest2,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTest3 = xTest[5]
digitTest3 = np.array(digitTest3)
digitTest3 = digitTest3.reshape((28,28))
plt.imshow(digitTest3,cmap=plt.cm.binary)
plt.show()

figure = plt.figure(figsize=(15,8))
digitTest4 = xTest[500]
digitTest4 = np.array(digitTest4)
digitTest4 = digitTest4.reshape((28,28))
plt.imshow(digitTest4,cmap=plt.cm.binary)
plt.show()

MODEL

# model training
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
lj = LogisticRegression(solver="liblinear").fit(xTrain,yTrain)
gnb = GaussianNB().fit(xTrain,yTrain)
knnc = KNeighborsClassifier().fit(xTrain,yTrain)
cartc = DecisionTreeClassifier(random_state=42).fit(xTrain,yTrain)
rfc = RandomForestClassifier(random_state=42,verbose=False).fit(xTrain,yTrain)

# model list for loop, you can do it one by one if you want

modelsc = [lj,gnb,knnc,cartc,rfc]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(xTest)
    R2CV = cross_val_score(model,xTest,yTest,cv=10,verbose=False).mean()
    error = -cross_val_score(model,xTest,yTest,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print("ACC-->",accuracy_score(yTest,predict))
    print("-" * 10)
    print("R2CV-->",R2CV)
    print("-" * 10)
    print("MEAN SQUARED ERROR-->",np.sqrt(error))
    print("-" * 30)

as we see, the best is Random Forest / R2 - %78.888 we need to be tuning

RANDOM FOREST PROCESS

CREATING NEW RANDOM FOREST MODEL

rfcmodel = RandomForestClassifier(max_depth=5,
                                 max_features=5,
                                 n_estimators=500,
                                 min_samples_split=10,
                                 verbose=False,
                                 random_state=42).fit(xTrain,yTrain)
PREDICTION FOR TUNED MODEL

predictrfcmodel = rfcmodel.predict(xTest)
CONTROLLING ACCURACY

R2CVtuned = cross_val_score(rfcmodel,xTest,yTest,cv=10,verbose=False).mean()
print(R2CVtuned)

CONTROLLING CONFUSION MATRIX

conf = confusion_matrix(yTest,predictrfcmodel)
# ıt is for visualization of confusion matrix

figure = plt.figure(figsize=(15,8))
sns.heatmap(conf,annot=True,cmap="PiYG",linewidths=2, linecolor='black')
plt.show()

PREDICTION FOR MAIN RANDOM FOREST

testLabel = pd.read_csv("/content/drive/MyDrive/fashion-mnist_test.csv")
testLabel = testLabel.drop("label",axis=1)

# Drop the rows that have `NaN` values
testLabel.dropna(inplace=True)

predictionRandom = rfc.predict(testLabel)

print(predictionRandom[1])

figure = plt.figure(figsize=(15,8))
TestPic = testLabel.iloc[1]
TestPic = np.array(TestPic)
TestPic = TestPic.reshape((28,28))
plt.imshow(TestPic,cmap=plt.cm.binary)
plt.show()

6 means shirt, but prediction is not true it is for example

ANN - ARTIFICIAL NEURAL NETWORK

xN = dataTrain.drop("label",axis=1)
yN = dataTrain["label"]

print(xN.shape)

xNTrain,xNTest,yNTrain,yNTest = train_test_split(xN,yN,test_size=0.2,random_state=42)

print(xNTrain.shape)

print(dataTrain["label"].value_counts())

MODEL PROCESS

ANNmodel = tf.keras.models.Sequential([
  # inputs
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # hiddens layers
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  # output layer
  tf.keras.layers.Dense(10)
])

lossfunc = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

ANNmodel.compile(optimizer='adam', loss=lossfunc, metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

MainModel = ANNmodel.fit(xNTrain, yNTrain, validation_split=0.2, epochs=30, callbacks=[callback],validation_data=(xNTest,yNTest))

ACCURACY CHECKING

print(ANNmodel.summary())

HistoryData = pd.DataFrame(ANNmodel.history.history)

print(HistoryData.head())

HistoryData.plot()

plt.plot(MainModel.history["accuracy"])
plt.plot(MainModel.history["val_accuracy"])
plt.ylabel("ACC")
plt.legend()
plt.show()

HistoryDict = MainModel.history

print(HistoryDict.keys())

val_losses = HistoryDict["val_loss"]
val_acc = HistoryDict["val_accuracy"]
acc = HistoryDict["accuracy"]
losses = HistoryDict["loss"]
epochs = range(1,len(val_losses)+1)

plt.plot(epochs,val_losses,"bo",label="LOSS")
plt.plot(epochs,val_acc,"r",label="ACCURACY")
plt.title("LOSS & ACCURACY")
plt.xlabel("EPOCH")
plt.ylabel("Loss & Acc")
plt.legend()
plt.show()

plt.plot(epochs,acc,"bo",label="ACCURACY")
plt.plot(epochs,val_acc,"r",label="ACCURACY VAL")
plt.title("ACCURACY & ACCURACY VAL")
plt.xlabel("EPOCH")
plt.ylabel("ACCURACY & ACCURACY VAL")
plt.legend()
plt.show()

plt.plot(epochs,losses,"bo",label="LOSS")
plt.plot(epochs,val_losses,"r",label="LOSS VAL")
plt.title("LOSS & LOSS VAL")
plt.xlabel("EPOCH")
plt.ylabel("LOSS & LOSS VAL")
plt.legend()
plt.show()

PREDICTION

print(dataTest.shape)

dataTest = np.array(dataTest)

print(dataTest.shape)

predictANN = ANNmodel.predict(dataTest)

p = [np.argmax(i) for i in predictANN]

print(p[10])

3 means dress, prediction is true, check content again

Import packages

#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from xgboost import XGBClassifier

import time

#from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
Load dataset

train = pd.read_csv('/content/drive/MyDrive/fashion-mnist_train.csv')
test = pd.read_csv('/content/drive/MyDrive/fashion-mnist_test.csv')

df_train = train.copy()
df_test = test.copy()

df_test.isnull().any().sum()

Examine NaN values

df_train.isnull().any().sum()

df_test.isnull().any().sum()

# Drop the rows that have `NaN` values
df_train.dropna(inplace=True)

print(df_train.duplicated().sum())
# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

# Drop the rows that have `NaN` values
df_test.dropna(inplace=True)

print(df_test.duplicated().sum())
# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

Separating data and label

X_train= df_train.drop(['label'],axis = 1)
X_train

X_train.shape

X_test = df_train['label']
X_test

X_test.shape

y_test = df_test.drop(['label'],axis = 1)
y_test.shape

Normalization

X_train = X_train.astype('float32')
y_test = y_test.astype('float32')
X_train /= 255.0
y_test /=255.0

Split training and test sets

seed = 99
np.random.seed(seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, X_test, test_size=0.1, random_state = seed)
Dimensionality Reduction using PCA

pca = PCA(n_components=100, random_state=42)
X_train_pca =pca.fit_transform(X_train)
X_test_pca = pca.transform(X_val)
y_test_pca =pca.transform(y_test)

X_train_pca.shape

X_train_PCA1 = pd.DataFrame(X_train_pca)
X_test_PCA1 = pd.DataFrame(X_test_pca)
Evaluate the model

The algorithms which we are using for classification and analysis of the data are:

1) Logistic Regression

2) SVM

3) Random Forest

4) Gradient Boosting

5) XGBoost

# 1. LR Model
start1 = time.time()

logistic = LogisticRegression(max_iter=200, solver='liblinear')
logistic.fit(X_train_PCA1, y_train)

end1 = time.time()
lr_time = end1-start1

# 2. SVC Model
start2 = time.time()

svc = SVC(C=13,kernel='rbf',gamma="auto",probability = True)
svc.fit(X_train_PCA1, y_train)

end2 = time.time()
svm_time = end2-start2

# 3. Random Forest
start3 = time.time()

random_forest = RandomForestClassifier(criterion='entropy', max_depth=70, n_estimators=100)
random_forest.fit(X_train_PCA1, y_train)
end3 = time.time()
forest_time = end3-start3

# 4. Gradient Boosting Method
start4 = time.time()

Gradient = ensemble.GradientBoostingClassifier(n_estimators=100)
Gradient.fit(X_train_PCA1, y_train)

end4 = time.time()
gradient_time = end4-start4

# 5. XGBoost Method
start5 = time.time()

xgb = XGBClassifier(use_label_encoder=False,objective="multi:softmax",eval_metric="merror")
xgb.fit(X_train_PCA1, y_train.ravel())

end5 = time.time()
xgb_time = end5-start5

print("LR Time: {:0.2f} minute".format(lr_time/60.0))
print("SVC Time: {:0.2f} minute".format(svm_time/60.0))
print("Random Forest Time: {:0.2f} minute".format(forest_time/60.0))
print("Gradient Boosting Time: {:0.2f} minute".format(gradient_time/60.0))
print("XGBoost Time: {:0.2f} minute".format(xgb_time/60.0))

Predicting the models

Logistic Regression Report and Analysis

y_train_lr = logistic.predict(X_train_PCA1)
y_pred_lr = logistic.predict(X_test_pca)
logistic_train = metrics.accuracy_score(y_train,y_train_lr )
logistic_accuracy = metrics.accuracy_score(y_val, y_pred_lr)

print("Train Accuracy score: {}".format(logistic_train))
print("Test Accuracy score: {}".format(logistic_accuracy))
print(metrics.classification_report(y_val, y_pred_lr))

con_matrix = pd.crosstab(pd.Series(y_val.values.flatten(), name='Actual' ),pd.Series(y_pred_lr, name='Predicted'))
plt.figure(figsize = (9,6))
plt.title("Confusion Matrix on Logistic Regression")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt='g')
plt.show()

SVM Report and Analysis

y_train_svc = svc.predict(X_train_PCA1)
y_pred_svc = svc.predict(X_test_pca)
svc_train = metrics.accuracy_score(y_train,y_train_svc)
svc_accuracy = metrics.accuracy_score(y_val, y_pred_svc)

print("Train Accuracy score: {}".format(svc_train))
print("Test Accuracy score: {}".format(svc_accuracy))
print(metrics.classification_report(y_val, y_pred_svc))

con_matrix = pd.crosstab(pd.Series(y_val.values.flatten(), name='Actual' ),pd.Series(y_pred_svc, name='Predicted'))
plt.figure(figsize = (9,6))
plt.title("Confusion Matrix on SVC")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt='g')
plt.show()

Random Forest Report and Analysis

y_train_forest = random_forest.predict(X_train_PCA1)
y_pred_forest = random_forest.predict(X_test_pca)
random_forest_train = metrics.accuracy_score(y_train,y_train_forest)
random_forest_accuracy = metrics.accuracy_score(y_val, y_pred_forest)

print("Train Accuracy score: {}".format(random_forest_train))
print("Test Accuracy score: {}".format(random_forest_accuracy))
print(metrics.classification_report(y_val, y_pred_forest))

con_matrix = pd.crosstab(pd.Series(y_val.values.flatten(), name='Actual' ),pd.Series(y_pred_forest, name='Predicted'))
plt.figure(figsize = (9,6))
plt.title("Confusion Matrix on Logistic Regression")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt='g')
plt.show()

Gradient Boosting Report and Analysis

y_train_gradient = Gradient.predict(X_train_PCA1)
y_pred_gradient = Gradient.predict(X_test_pca)
gradient_train = metrics.accuracy_score(y_train,y_train_gradient)
gradient_accuracy = metrics.accuracy_score(y_val, y_pred_gradient)

print("Train Accuracy score: {}".format(gradient_train))
print("Test Accuracy score: {}".format(gradient_accuracy))
print(metrics.classification_report(y_val, y_pred_gradient))

con_matrix = pd.crosstab(pd.Series(y_val.values.flatten(), name='Actual' ),pd.Series(y_pred_gradient, name='Predicted'))
plt.figure(figsize = (9,6))
plt.title("Confusion Matrix on Logistic Regression")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt='g')
plt.show()

XGBoost Report and Analysis

y_train_xgboost = xgb.predict(X_train_PCA1)
y_pred_xgboost = xgb.predict(X_test_pca)
xgb_train = metrics.accuracy_score(y_train,y_train_xgboost)
xgb_accuracy = metrics.accuracy_score(y_val, y_pred_xgboost)

print("Train Accuracy score: {}".format(xgb_train))
print("Test Accuracy score: {}".format(xgb_accuracy))
print(metrics.classification_report(y_val, y_pred_xgboost))

con_matrix = pd.crosstab(pd.Series(y_val.values.flatten(), name='Actual' ),pd.Series(y_pred_xgboost, name='Predicted'))
plt.figure(figsize = (9,6))
plt.title("Confusion Matrix on Logistic Regression")
sns.heatmap(con_matrix, cmap="Blues", annot=True, fmt='g')
plt.show()

Model Comparison

Train_Accuracy = [logistic_train,svc_train,random_forest_train,gradient_train,xgb_train]
Test_Accuracy = [logistic_accuracy,svc_accuracy,random_forest_accuracy,gradient_accuracy,xgb_accuracy]
data1 = {
    'Algorithm': ['Logistic Regression','SVC','Random Forest Classifier','Gradient Boosting','XGBoost'],
    'Train Accuracy':Train_Accuracy,
    'Test Accuracy':Test_Accuracy
}

df1 = pd.DataFrame(data1)

df1

fig = go.Figure(data=[
    go.Bar(name='train set', x=data1['Algorithm'], y=data1['Train Accuracy'],text=np.round(data1['Train Accuracy'],2),textposition='outside'),
    go.Bar(name='test set', x=data1['Algorithm'], y=data1['Test Accuracy'],text=np.round(data1['Test Accuracy'],2),textposition='outside')
])

fig.update_layout(barmode='group',title_text='Accuracy Comparison On Different Models',yaxis=dict(
        title='Accuracy'))
fig.show()

Conclusion

Computation Time**
Gradient Boost>XGBoost>SVC>Random Forest>Logistic Regeression

Train Accuracy
Random Forest>XGBoost>Gradient Boost>SVC>Logistic Regression

Test Accuracy
SVC>XGBoost>Random Forest>Gradient Boost>Logistic Regression

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.image as mpimg

# Data Preprocessing:

from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,roc_curve, roc_auc_score, accuracy_score
# Models:

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

# A model that I learned by myself: CatBoost + Plotly

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# offline (for plotly)
import plotly.offline as pyo

# Clustering:

from sklearn.cluster import KMeans

# PCA:

from sklearn.decomposition import PCA

# ICA:

from sklearn.decomposition import FastICA

# Scaling:

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Cross Validation:

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# training df:

df_train = pd.read_csv("/content/drive/MyDrive/fashion-mnist_train.csv") # csv pandas df

df_train.sample(n = 4, random_state = 8).sort_values(by = 'label')

# testing df: we will not train our model on this df, we will test out model on it at the end of the notebook

df_test = pd.read_csv("/content/drive/MyDrive/fashion-mnist_test.csv") # csv pandas df

df_test.sample(n = 4, random_state = 8).sort_values(by = 'label')

df_train.shape # 60000 rows and 785 columns (784 pixels + label column)

df_test.shape # 10000 rows and 785 columns (784 pixels + label column) - test

df_train.describe()

As you can see, each label can be taken on its own. It is also possible to notice that in general, there are some differences between the labels.

df_train[df_train['label']==3].describe() # 3 = Dress

df_train[df_train['label']==9].describe() # 9 = Ankle boot

df_train.isnull().sum().sum() # the dataset has no NaN values

df_train.label.unique() # we have 0-9 labels: 10 labels

# Drop the rows that have `NaN` values
df_train.dropna(inplace=True)

print(df_train.duplicated().sum())
# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

# Drop the rows that have `NaN` values
df_test.dropna(inplace=True)

print(df_test.duplicated().sum())
# since we are dealing with pixels, this section should be noted, duplicated values can be important for data, so we will not do anything

cor = df_train.corr() # it calculates the correation between each two features
Since the feature that interests us is the label, we would like to see only the correlation between each feature and the label.

# selecting highly correlated features

relevant_features = cor_target[cor_target > 0.5] # shows only corr with 0.5 or higher value

# 105 features is nice(106 - label), but I would like to know about higher corr than 0.5

for i in range(51):
        print("Higher than",round(0.5 + i*0.01, 2), "correlation: # of Pixels:",len(cor_target[cor_target > (0.5 + i*0.01)])-1)
        if len(cor_target[cor_target > (0.5 + i*0.01)]) == 1:
            break

In summary, it can be seen that there are around 36 pixels that have a significant effect on the label of each image (more than 0.6). We will consider using this value for the amount of pixels we will take from each image. On the other hand, it will be interesting to see if dimensionality reduction algorithms (PCA or ICA) will give us the same amount of features

target = df_train['label'] # the feature we would like to predict, the label of picture
data = df_train.drop(['label'], axis = 1) # we will drop y from x, because we want to predict it

# we will split our testing dataset into data & target. We can't train this dataset. we will use it as a X_test & y_test

test_labels = df_test['label'] # the feature we would like to predict, the label of picture
test = df_test.drop(['label'], axis = 1) # we will drop y from x, because we want to predict it
There are 10 labels, it means that there are 10 different types of clothing to be classified:

0 - T-Shirt/Top

1 - Trouser

2 - Pullover

3 - Dress

4 - Coat

5 - Sandal

6 - Shirt

7 - Sneaker

8 - Bag

9 - Ankle Boot

# A function to show the labels
def num_to_name(label):
    labeled = label.copy()
    mapping = {0 :'T-Shirt/Top',
    1 :'Trouser',
    2 :'Pullover',
    3 :'Dress',
    4 :'Coat',
    5 :'Sandal',
    6 :'Shirt',
    7 :'Sneaker',
    8 :'Bag',
    9 :'Ankle Boot'}
    labeled = label.map(mapping)
    return labeled

# exaple of pictures with their correct label

fig, axes = plt.subplots(4, 4, figsize = (12,12))
axes = axes.ravel()

for i in range(16):
    axes[i].imshow(data.values.reshape((data.shape[0], 28, 28))[i], cmap=plt.get_cmap('binary'))
    axes[i].set_title("Outfit " + str(target[i]) + ": "+ num_to_name(target)[i])
    axes[i].axis('off')
plt.show()

target.value_counts() # how much exapmles we have from each label

# a simple counter graph for it

plt.subplots(figsize = (15,5))
plt.title("Outfits Counter", size=20)
fig = sns.countplot(num_to_name(target))

The data is very balanced in terms of the amount of samples we have from each label. We will conclude unequivocally that a dummy classifier model will give us a 10% success rate for ten labels. This is a very low percentage that will also be very easy to pass.

Data Processing

The things that are important to consider before start to work with the data:

As we can see, when a pixel value is 0 (black pixel), it means that its an empty pixel. We would like to drop pixels which are empty in most of the pictures. We will consider choosing to use one of the following methods in order to make our model more compact: PCA and K-means. Note that in the PCA method we would like to use as few pixels as possible in order to label the item of clothing. When using the K-Means method, we want to use as few colors as possible in the classification process. We would like to understand which labels are more difficult to classify, and which are easier. We may consider using a more complex model in cases where the differences between two labels are the most minor and requires a maximum of data. On the other hand, when an item is easy to identify, less data is required to classify it.

Training / Testing Split:

X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, random_state=18)
X_test = test.copy() # since we already split test_df into data and labels, we cant do more actions on it
Scaling:

# important for infinty values cases

X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)
Since we have 256 pixels (0-255), we can divide them into 255 in order to achieve the desired range values (0-1).¶

X_train = X_train / 255
X_val = X_val / 255
X_test = X_test / 255
Dimensionality Reduction

PCA (Principal Component Analysis)

Principal Component Analysis (PCA) is a classical technique in statistical data analysis, feature extraction and data reduction, aiming at explaining observed signals as a linear combination of orthogonal principal components.

We would like to find our optimal n_components value

pca = PCA() # all 784 features
pca.fit(X_train)

# A graph to present the conection between the num of features & the explained variance:
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

fig = px.area(
    title = "Explained variance as a function of the number of dimensions:",
    x=range(1, exp_var_cumul.shape[0] + 1),
    y=exp_var_cumul * 100,
    labels={"x": "# of Pixels", "y": "Explained Variance"},
    width = 1000 ,
    height = 500
)

fig.show()

We can see that by taking only 24 features, we stay with 80% of explained variance.

pca = PCA(n_components=0.80) # we can try using svd_solver="randomized"
X_train_reduced = pca.fit_transform(X_train)
X_val_reduced = pca.transform(X_val)
pca.n_components_

# A three-dimensional graph depicting the way our data is interpreted, plotly does it easily for us
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter_3d(
    X_train_reduced, x=0, y=1, z=2, color = num_to_name(y_train),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()

# 2D version: with x and y
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(
    X_train_reduced, x=0, y=1, color = num_to_name(y_train),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2'}
)
fig.show()

# 2D version: with y and z
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(
    X_train_reduced, x=1, y=2, color = num_to_name(y_train),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2'}
)
fig.show()

# 2D version: with z and y
total_var = pca.explained_variance_ratio_.sum() * 100
fig = px.scatter(
    X_train_reduced, x=2, y=1, color = num_to_name(y_train),
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2'}
)
fig.show()

ICA (Independent Component Analysis)

Independent Component Analysis (ICA) is a technique of array processing and data analysis, aiming at recovering unobserved signals or ‘sources’ from observed mixtures, exploiting only the assumption of mutual independence between the signals. The separation of the sources by ICA has great potential in applications such as the separation of sound signals (like voices mixed in simultaneous multiple records, for example), in telecommunication or in the treatment of medical signals. However, ICA is not yet often used by statisticians. While the goal in PCA is to find an orthogonal linear transformation that maximizes the variance of the variables, the goal of ICA is to find the linear transformation, which the basis vectors are statistically independent and non-Gaussian

For more info about how ICA is different from PCA: http://www2.hawaii.edu/~kyungim/papers/baek_cvprip02.pdf

I chose to use ICA instead of PCA because the pictures in this dataset are very neat (as opposed to a dataset of dogs vs cats). ICA algorithm is faster and I think that in this specific case, using a simpler algorithm will actually give better results, and more importantly, in less time. Also, we have already received from PCA's analysis the amount of pixels needed, so I will just put 24 pixels into the ICA algorithm

ica = FastICA(n_components=24, random_state=18) # I took the results from PCA and applied them on ICA (24 pixels)
X_train_reduced = ica.fit_transform(X_train) # fit ica on train
X_val_reduced = ica.transform(X_val) # aplly ica on validation
X_train_reduced.shape

Starting in v1.3, whiten='unit-variance' will be used by default.

X_train = pd.DataFrame(X_train_reduced)
X_val = pd.DataFrame(X_val_reduced)
Models

We will use some of the models we have learned trought this year to predict the type of the outfit. We would like to use the three best model to create a voting model.

Naive Bayes

This model is extremly simple. We will use it as our dummy model. Dummy Classifier should do 10% of success in a multi class classification of 10 equal classes.

bayes = GaussianNB()
bayes.fit(X_train, y_train)
bayes

y_pred = bayes.predict(X_val)
bayes_acc = accuracy_score(y_val, y_pred)
bayes_acc

print (classification_report(y_val, y_pred))

Due to Guessian Naive Bayes results, our model must be better than 73% of success on our validation test.

KNN

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn

y_pred = knn.predict(X_val)
knn_acc = accuracy_score(y_val, y_pred)
knn_acc

print (classification_report(y_val, y_pred))

Logistic Regression

lr = LogisticRegression(solver = 'lbfgs')
lr.fit(X_train, y_train)
lr

y_pred = lr.predict(X_val)
lr_acc = accuracy_score(y_val, y_pred)
lr_acc

print (classification_report(y_val, y_pred))

CatBoost

!pip install catboost

from catboost import CatBoostClassifier
cat = CatBoostClassifier(logging_level='Silent')
cat.fit(X_train, y_train)
cat

y_pred = cat.predict(X_val)
y_pred_cat = y_pred
cat_acc = accuracy_score(y_val, y_pred)
cat_acc

print (classification_report(y_val, y_pred))

AdaBoost

rfc = RandomForestClassifier(n_estimators=10)
ada = AdaBoostClassifier(n_estimators=100,learning_rate= 0.1, base_estimator=rfc)
ada.fit(X_train, y_train)
ada

y_pred = ada.predict(X_val)
ada_acc = accuracy_score(y_val, y_pred)
ada_acc

print (classification_report(y_val, y_pred))

XGBoost

xgb = XGBClassifier(use_label_encoder =False)
xgb.fit(X_train, y_train)
xgb

y_pred = xgb.predict(X_val)
xgb_acc = accuracy_score(y_val, y_pred)
xgb_acc

print (classification_report(y_val, y_pred))

Random Forest

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf

y_pred = rf.predict(X_val)
rf_acc = accuracy_score(y_val, y_pred)
rf_acc

print (classification_report(y_val, y_pred))
              precision    recall  f1-score   support

           0       0.76      0.85      0.80        55
           1       0.95      0.97      0.96        59
           2       0.69      0.77      0.73        56
           3       0.73      0.80      0.76        50
           4       0.85      0.72      0.78        65
           5       0.77      0.85      0.81        48
           6       0.60      0.46      0.53        56
           7       0.86      0.83      0.84        69
           8       0.87      0.84      0.85        49
           9       0.88      0.93      0.90        61

    accuracy                           0.80       568
   macro avg       0.80      0.80      0.80       568
weighted avg       0.80      0.80      0.80       568

Now we would like to take our top 3 models and mix them into a voting model:

Voting

Soft Voting/Majority Rule classifier for unfitted estimators.

Hard Voting:

clf1 = xgb
clf2 = knn
clf3 = rf
clf4 = cat

hv = VotingClassifier(estimators=[
        ('xgb', clf1), ('knn', clf2), ('rf', clf3)], voting='hard')
hv.fit(X_train, y_train)
hv

y_pred = hv.predict(X_val)
hv_acc = accuracy_score(y_val, y_pred)
hv_acc

print (classification_report(y_val, y_pred))
  
print (classification_report(y_val, y_pred))

Soft Voting:

sv = VotingClassifier(estimators=[
        ('xgb', clf1), ('cat', clf4), ('rf', clf3)], voting='soft', weights=[1,3,1])
sv.fit(X_train, y_train)
sv

y_pred = sv.predict(X_val)
y_pred_sv = y_pred.copy()
sv_acc = accuracy_score(y_val, y_pred)
sv_acc

print (classification_report(y_val, y_pred))
      
It is clear that our attempt to unify the models was unsuccessful. We will continue with CatBoost

Conclusion:

acc_list = {'Model':  ['Naive Bayes', 'KNN','Logistic Regression','CatBoost', 'AdaBoost', 'XGBoost','Random Forest','Hard Voting', 'Soft Voting'],
        'Accuracy': [bayes_acc,knn_acc,lr_acc,cat_acc,ada_acc,xgb_acc,rf_acc,hv_acc,sv_acc],
        }

fig = go.Figure(data=[
    go.Bar(name='train set', x=acc_list['Model'], y=acc_list['Accuracy'],text=np.round(acc_list['Accuracy'],2),textposition='outside'),
])
fig.update_layout(barmode='group',title_text='Accuracy Comparison On Different Models',yaxis=dict(
        title='Accuracy'))
fig.show()

cm = confusion_matrix(y_val, y_pred_cat)
plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest', cmap = plt.cm.coolwarm)
plt.title('Confusion matrix for CatBoost', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ['T-shirt/top', 'Trouser', 'Pullover',
                        'Dress', 'Coat', 'Sandal', 'Shirt',
                        'Sneaker', 'Bag', 'Ankle boot'], rotation=45, size = 10)
plt.yticks(tick_marks, ['T-shirt/top', 'Trouser', 'Pullover',
                        'Dress', 'Coat', 'Sandal', 'Shirt',
                        'Sneaker', 'Bag', 'Ankle boot'], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x),
        horizontalalignment='center',
        verticalalignment='center')

Conclusion

Computation Time**
Gradient Boost>XGBoost>SVC>Random Forest>Logistic Regeression

Train Accuracy
Random Forest>XGBoost>Gradient Boost>SVC>CNN>Logistic Regression

Test Accuracy
CNN>Soft voting>hard voting>Cat Boost>Ada Boost>KNN>ANN>XGBoost>SVC>Random Forest>Logistic Regression>Gradient Boosting>Naive Bais> DecisionTreeClassifier>GaussianNB

The Best Model is CatBoost, with 84.3%% of success, by using only 24 pixels
