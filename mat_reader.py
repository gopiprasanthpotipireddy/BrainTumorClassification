import os
import h5py as h5
import numpy as np
import pandas as pd
import mat_img_reader as mir
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


f=[]
file_path="C:/Users/HOME/Desktop/PRASANTH/BrainTumourClassification/1512427"

for i in range(3064):
     f.append(h5.File(os.path.join(file_path,str(i+1)+".mat"),'a'))
     

 

"""
f=h5.File(os.path.join(file_path,"10.mat"),'a')
  for k,v in f.items():
    np.array(k)
    np.array(v)
   
"""

p=[]
for i in range(3064):
    p.append(mir.Patient('','','','',''))
    p[i].image=np.array(f[i]['/cjdata/image'])
    p[i].PID=np.array(f[i]['/cjdata/PID'])
    p[i].label=f[i]['/cjdata/label'][0][0]
    p[i].tumorBorder=f[i]['/cjdata/tumorBorder'][0]
    p[i].tumorMask=list(f[i]['/cjdata/tumorMask'])[0]

columns =['PID', 'image', 'label', 'tumorBorder', 'tumorMask']    

d={'PID':[], 'image':[], 'label':[], 'tumorBorder':[], 'tumorMask':[]}

for i in range(1000):
    d['PID'].append(p[i].PID)
    d['image'].append(p[i].image)
    d['label'].append(p[i].label)
    d['tumorBorder'].append(p[i].tumorBorder)
    d['tumorMask'].append(p[i].tumorMask)
    

#making dataframe
Patient_data=pd.DataFrame(list(d.values()),columns)

Patient_data=Patient_data.transpose()

#preprocessing image datafor training x_train


#files with correpted images
count=0
for i in range(3064): 
     if np.array(p[i].image).shape[0]!= 512 or np.array(p[i].image).shape[1]!=512:
         print("worst")
         print(i)
         count+=1

#training images only having (512,512)
image_train=[]
label_train=[]
         
for i in range(1207,3064):
    image_train.append(p[i].image)
    label_train.append(p[i].label)



x_train=np.array(image_train)
label_train=np.array(d['label'])
np.concatenate(x_train,np.array(image_train))

np.mat(d['image']).shape

#feature reduction  formula X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T



image_train= image_train.reshape(image_train.shape[0],image_train.shape[1]*image_train.shape[2])

#label for y_train
label_train=np.array(d['label'])
label_train=label_train.reshape(label_train.shape[0],)

image_train, label_train = shuffle(image_train, label_train, random_state=42)

Tumour_classifier = tree.DecisionTreeClassifier()

#testing the model


X_train, X_test, y_train, y_test = train_test_split(image_train, label_train, test_size=0.2, random_state=42)
Tumour_classifier.fit(X_train, y_train)

preds = Tumour_classifier.predict(X_test)


print("Accuracy:", accuracy_score(y_test,preds))


#image plotting from (512,512) matrix
plt.imshow(image_train[1])




