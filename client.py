import seaborn as sns
import lime
import shap
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import preprocessing 

import flwr as fl

df= pd.read_csv("I:\My_Project\healthcare-dataset-stroke-data.csv")


nan_values= df.isna()
nan_count = nan_values.sum()
print(nan_count)

df= df.dropna()

print("The number of nan values are : ",df.isna())
df=df.drop_duplicates()

print("The number of duplicated values are : ", df.duplicated().sum())


label_encoder = preprocessing.LabelEncoder()
df['stroke']=label_encoder.fit_transform(df['stroke'])
df['smoking_status']=label_encoder.fit_transform(df['smoking_status'])
df['Residence_type']=label_encoder.fit_transform(df['Residence_type'])
df['work_type']=label_encoder.fit_transform(df['work_type'])
df['gender']=label_encoder.fit_transform(df['gender'])
df['ever_married']=label_encoder.fit_transform(df['ever_married'])


df= df.drop(['id'],axis=1)



columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']

sl = StandardScaler()
columns_to_scale = ['age','avg_glucose_level','bmi']
df[columns_to_scale]= sl.fit_transform(df[columns_to_scale])
X=df.drop('stroke',axis=1)
Y=df['stroke']
X_train, X_test,y_train, y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
sm=SMOTE(random_state=42,sampling_strategy="auto",n_jobs=-1)
X_res,Y_res=sm.fit_resample(X_train,y_train)


model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_res.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16, activation='relu'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    model.set_weights(parameters)
    model.fit(X_res, Y_res, epochs=100, batch_size=32)
    return model.get_weights(), len(X_res), {}

  def evaluate(self, parameters, config):
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(X_test, y_test)
    return loss, len(X_test), {"accuracy": accuracy}


fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())


#model.fit(X_res, Y_res, epochs=100, batch_size=32)

'''loss, accuracy = model.evaluate(X_test, y_test)

print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

y_predict=model.predict(X_test)

y_pred_classes = np.round(y_predict)
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()'''

