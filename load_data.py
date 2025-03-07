import numpy as np
import os.path
from pathlib import Path
import scipy.io as sio

def load_data(name):
    if name == 'IP':
        path = 'Datasets/Indian_pines/Indian_pines_corrected.mat'
        T =sio.loadmat(path)['indian_pines_corrected']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Indian_pines/Indian_pines_gt.mat')['indian_pines_gt']

##############################################################################
    elif name == 'PU':
        path = 'Datasets/Pavia_University/PaviaU.mat'
        T =sio.loadmat(path)['paviaU']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/Pavia_University/PaviaU_gt.mat')['paviaU_gt'] 
##############################################################################

    elif name == 'HOU13':
        path = 'Datasets/Houston13/Houstondata.mat'
        T =sio.loadmat(path)['Houstondata']
        T = T.astype(np.float32)
        
        labels = sio.loadmat('Datasets/Houston13/Houstonlabel.mat')['Houstonlabel'] 

##############################################################################
    elif name == 'KSC':
        path = 'Datasets/KSC/KSC.mat'
        T =sio.loadmat(path)['KSC']
        T = T.astype(np.float32)
        labels = sio.loadmat('Datasets/KSC/KSC_gt.mat')['KSC_gt'] 

##############################################################################
    else:
        print("Incorrect data name")
        
    return T, labels