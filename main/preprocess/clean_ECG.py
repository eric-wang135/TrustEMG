import os, numpy as np
import wfdb
from util import *
from tqdm import tqdm
from scipy import signal as sig

def get_ecg_filepaths(directory):
    ecg_id=[16265,16272,16273,16420,16483,16539,16773,16786,16795,17052,17453,18177,18184,19088,19090,19093,19140,19830]
    ecg_paths =[]
    for i in range(len(ecg_id)):
        ecg_paths.append(os.path.join(directory, str(ecg_id[i])))
    return ecg_paths

def read_ecg(ecg_path):
    
    ecg = wfdb.rdrecord(ecg_path).__dict__.get('p_signal')[:,0] #channel 1 ECG, wfdb.rdrecord(filepath,sampto=1000)
    ecg_rate = 1000
    ecg = resample(ecg,128,ecg_rate)
    ecg = ecg.astype('float64')
    return ecg

Corpus_path = '../mit-bih-normal-sinus-rhythm-database-1.0.0'
out_path ='../ECG'

check_path(out_path+'_train')
check_path(out_path+'_test')

file_paths = get_ecg_filepaths(Corpus_path)

# Filtering possible noise
b_h, a_h = sig.butter(3, 1, 'hp',fs=1000)
notch = sig.iirnotch(60, 35, fs=1000)
b_l, a_l = sig.butter(3, 200, 'lp',fs=1000)


start,end = 40000, 110000 # ECG Segment of 70 seconds
train_num = 14 # number of data in training set

for i in tqdm(range(len(file_paths))):
    ecg_file = read_ecg(file_paths[i])
    
    if i<train_num:
        save_path = out_path+'_train'
    else:
        save_path = out_path+'_test'

    if ecg_file.ndim>1:
        for j in range(ecg_file.shape[1]):
            ecg_save = sig.filtfilt(b_l,a_l,sig.filtfilt(b_h,a_h,ecg_file[start:end,j]))
            np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+str(j)),ecg_save)
    else:
        ecg_file = sig.filtfilt(b_l,a_l,sig.filtfilt(b_h,a_h,ecg_file[start:end]))
        np.save(os.path.join(save_path,'E'+file_paths[i].split(os.sep)[-1].split(".")[0]),ecg_file)
    
    