import os, argparse, random, numpy as np
import wfdb
from scipy import signal as sig
import scipy.io
from itertools import combinations
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='PLI') # type: BW/PLI/ECG/MOA/QM/WGN
    parser.add_argument('--mix_num', type=int, default=1)
    parser.add_argument('--path', type=str, default='../sEMG_noise_train')
    parser.add_argument('--time_length', type=int, default=70)      # unit: second
    parser.add_argument('--sr', type=int, default=1000)             # sampling rate
    parser.add_argument('--number', type=int, default=1)            # number of the segments creating
    args = parser.parse_args()
    return args

def noise_sampleto(noise,sample_num):
    
    if noise.shape[0] < sample_num:
        tmp = (sample_num // noise.shape[0]) + 1
        y_noise = []
        for _ in range(tmp):
            y_noise.extend(noise)
    else:
        y_noise = noise
    
    y_noise = np.asarray(y_noise)
    start = random.randint(0,y_noise.shape[0]-sample_num)

    return y_noise[start:start+sample_num]

if __name__ == '__main__':
    
    args = get_args()
    all_noise_type = args.type.split('_')

    if args.mix_num > 1:
        # Mix noise
        sample_num = args.sr*args.time_length
        noise_type_lists = combinations(all_noise_type,args.mix_num)

        for noise_type_list in noise_type_lists:
            
            noise_path_list = []
            folder_name = ""
            
            for noise_type in noise_type_list:
                noise_list = get_filepaths(os.path.join(args.path,noise_type))
                
                if len(noise_list)< args.number:
                    noise_path_list.append(random.choices(noise_list,k=args.number))
                else:
                    noise_path_list.append(random.sample(noise_list,args.number))
                
                folder_name = folder_name+"+"+noise_type[0]
            
            out_path = os.path.join(args.path,folder_name[1:])
            check_path(out_path)
            
            for i in range(args.number):
                noise = []
                # Ramdomly cut noise and add them up
                noise_name = noise_path_list[0][i].split(os.sep)[-1].split(".")[0]
                noise1 = np.load(noise_path_list[0][i])
                    
                noise1 = noise_sampleto(noise1,sample_num)

                noise1_pw = np.dot(noise1,noise1)

                for n in range(1,len(noise_type_list)):
                    noise2 = np.load(noise_path_list[n][i])
                    noise2 = noise_sampleto(noise2,sample_num)
                    noise2_pw = np.dot(noise2,noise2)
                    noise2 = np.sqrt(noise1_pw/noise2_pw)*noise2
                    noise1 = noise1+noise2
                    
                    noise_name = noise_name+'+'+(noise_path_list[n][i].split(os.sep)[-1].split(".")[0])
                
                np.save(os.path.join(out_path,noise_name),noise1)

    else:
        time = np.linspace(0,args.time_length,args.sr*args.time_length,endpoint=False)
        out_path = os.path.join(args.path,args.type)
        check_path(out_path)
        if args.type == 'PLI':
            """
            f_center = 60
            f_dev = 1.5
            Train: 58.7, 58.9...61.2, 61.3; Test:  58.6, 59, 59.4, 59.8, 60.2, 60.6, 61, 61.4
            """
            f_step = 0.2 if 'train' in args.path else 0.4
            f_start = 58.7 if 'train' in args.path else 58.6

            for i in range(args.number):
                f = f_start + f_step * i
                PLI = np.sin(2*np.pi*f*time)
                np.save(os.path.join(out_path,args.type+"_"+str(i)),PLI)

        elif args.type == 'BW':
            record = wfdb.rdrecord('../mit-bih-noise-stress-test-database-1.0.0/bw')
            bw_sig =  record.__dict__.get('p_signal')
            ch =  0 if 'train' in args.path else 1 # Channel 1 for train and 2 for test

            BW = resample(bw_sig[:,ch],360,1000)
            segment_length = BW.shape[0]//args.number
            for n in range(args.number):
                np.save(os.path.join(out_path,"BW_ch"+str(ch)+"_"+str(n)),BW[n*segment_length:(n+1)*segment_length])
            
        elif args.type == 'QM':
            # Electord motion artifacts in NSTD is labeled as QM here
            record = wfdb.rdrecord('../mit-bih-noise-stress-test-database-1.0.0/em')
            sig =  record.__dict__.get('p_signal')
            ch =  0 if 'train' in args.path else 1 # Channel 1 for train and 2 for test

            # Time length: 30s/data
            EM = resample(sig[:,ch],360,1000)
            segment_length = EM.shape[0]//args.number
            for n in range(args.number):
                np.save(os.path.join(out_path,"QM_ch"+str(ch)+"_"+str(n)),EM[n*segment_length:(n+1)*segment_length])     

        elif args.type == 'WGN':
            mean = 0
            std = 1 
            for i in range(args.number):
                WGN = np.random.normal(mean,std,args.sr*args.time_length)
                np.save(os.path.join(out_path,"WGN_"+str(i)),WGN)

        elif args.type == 'MOA':
            paths = get_filepaths(os.path.join(args.path,args.type),ftype='.mat')
            notch = signal.iirnotch(60, 30, fs=2000)
            for path in paths:
                data = np.array(scipy.io.loadmat(path)['a']).squeeze()
                data = signal.filtfilt(notch[0],notch[1],data).astype('float64')
                data = np.convolve(data, np.ones(51)/51, mode='valid')[::2]
                np.save(path.replace('.mat','.npy'),data)

        """
        elif args.type == 'PLIhard+BW+ECG':
            
            noise_type_list = ["PLIhard","BW","ECG"]
            noise_path_list = []
            for noise_type in noise_type_list:
                noise_list = get_filepaths(os.path.join(args.path,noise_type))
                print(noise_list)
                noise_path_list.append(random.sample(noise_list,args.number))
            
            for i in range(args.number):
                noise = []
                noise_name = []
                noise1 = np.load(noise_path_list[0][i])
                noise_name.append(noise_path_list[0][i].split(os.sep)[-1].split(".")[0])
                # Ramdomly cut noise and add them up
                for n in range(1,len(noise_type_list)):
                    noise2 = np.load(noise_path_list[n][i])
                    if (noise2.shape[0]>noise1.shape[0]):
                        start = random.randint(0,noise2.shape[0]-noise1.shape[0])
                        end = start+noise1.shape[0]
                        noise1 = noise1+noise2[start:end]
                    else:
                        start = random.randint(0,noise1.shape[0]-noise2.shape[0])
                        end = start+noise2.shape[0]
                        noise1 = noise1[start:end]+noise2
                    
                    noise_name.append(noise_path_list[n][i].split(os.sep)[-1].split(".")[0])

                np.save(os.path.join(out_path,noise_name[0]+"+"+noise_name[1]+"+"+noise_name[2]),noise1)
        """  

