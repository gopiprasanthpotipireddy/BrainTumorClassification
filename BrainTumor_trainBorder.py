import os
import h5py as h5
import numpy as np
import pandas as pd
import mat_img_reader as mir
from sklearn import naive_bayes
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


f=[]
file_path="C:/Users/HOME/Desktop/PRASANTH/BrainTumourClassification/1512427"

for i in range(3064):
     f.append(h5.File(os.path.join(file_path,str(i+1)+".mat"),'a'))

#column names
f[0].name
list(f[0]['/'].keys())
list(f[0]['/cjdata'].keys())

    
d={'PID':[],
   'image':[],
   'label':[],
   'tumorBorder':[],
   'tumorMask':[]}
for i in range(3064):
    d['PID'].append(np.array(f[i]['/cjdata/PID']))
    d['image'].append(np.array(f[i]['/cjdata/image']))
    d['label'].append(f[i]['/cjdata/label'][0][0])
    d['tumorBorder'].append(f[i]['/cjdata/tumorBorder'][0])
    d['tumorMask'].append(f[i]['/cjdata/tumorMask'][0])


#files with corrupted images from (256,256) to (512,512)
count=0
for i in range(3064): 
     #if np.array(d['image'][i]).shape[0]!= 512 or np.array(d['image'][i]).shape[1]!=512:
         #print("images of 256*256 size")
         d['image'][i]=np.concatenate((np.zeros((256,256)),np.zeros((256,256))),axis=0)
         d['image'][i]=np.concatenate((d['image'][i],np.zeros((512,256))),axis=1)
         del f[i]['/cjdata/image']
         f[i]['/cjdata/image']=d['image'][i]
         count+=1
 #test
for i in range(3064): 
     if f[i]['/cjdata/image'].shape[0]!= 512 or f[i]['/cjdata/image'].shape[1]!=512:
         print("images of 256*256 size")

columns=['PID', 'image', 'label', 'tumorBorder', 'tumorMask']
Patient_data=pd.DataFrame(list(d.values()),columns)       
         
Patient_data=Patient_data.transpose()

Patient_data.to_csv(os.path.join(file_path,"BrainTumor_Data"))
#data=pd.read_csv(os.path.join(file_path,"BrainTumor_Data"))
