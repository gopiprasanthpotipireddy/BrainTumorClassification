import os
import h5py as h5
import numpy as np
import seaborn as sns
import pandas as pd

#parse each mat structure into a python patient object
class Patient(object):
    PID = ""
    image=""
    label=""
    tumorBorder=""
    tumorMask=""
    
    def __init__(self, PID, image, label,tumorBorder,tumorMask):
        self.PID = PID
        self.image = image
        self.label = label
        self.tumorBorder=tumorBorder
        self.tumorMask=tumorMask


file_path="C:/Users/HOME/Desktop/PRASANTH/brainTumorDataPublic_1766/"

#reading mat file using hdf reader
f=h5.File(os.path.join(file_path,"10.mat"),'a')

list(f.items()) #object keys /cjdata

#list of Obkect keys  ['PID', 'image', 'label', 'tumorBorder', 'tumorMask']
list(f['/cjdata'].keys())

p=Patient('','','','','')
p.image=np.mat(f['/cjdata/image'])
p.PID=np.array(f['/cjdata/PID'])
p.label=np.array(f['/cjdata/label'])
p.tumorBorder=np.mat(f['/cjdata/tumorBorder'])
p.tumorMask=np.mat(f['/cjdata/tumorMask'])

sns.heatmap(p.image)
sns.heatmap(p.tumorMask)



#making data set

    

        
