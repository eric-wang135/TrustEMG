import random, csv, os, argparse
import numpy as np
from tqdm import tqdm
from util import *
#import scipy.io, math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_path', type=str, default='../EMG_noise_train/')
    parser.add_argument('--noise_type', type=str, default='all')
    parser.add_argument('--dataset_type', type=str, default='train')
    parser.add_argument('--dir_end', type=str, default='_withSTI_seg2s')  #_withSTI_seg60s_E+P+B / _withSTI_seg60s_nsrd
    parser.add_argument('--cross_ch' , action='store_true', default=False)
    parser.add_argument('--segmentation' , type=int, default=-1)
    parser.add_argument('--noisy_num' , type=int, default=1)
    args = parser.parse_args()
    return args

def get_filepaths_withSTI(directory,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename[-5] !='i' and filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
    return sorted(file_paths)

def add_noise(clean_path, noise_path, SNR, return_info=False, normalize=False):
    clean_rate = 1000
    y_clean = np.load(clean_path)
    noise_ori = np.load(noise_path)
    
    #if noise shorter than clean wav, extend
    if len(noise_ori) < len(y_clean):
        tmp = (len(y_clean) // len(noise_ori)) + 1
        y_noise = []
        for _ in range(tmp):
            y_noise.extend(noise_ori)
    else:
        y_noise = noise_ori

    # Ramdomly cut noise
    
    start = random.randint(0,len(y_noise)-len(y_clean))
    end = start+len(y_clean)
    y_noise = y_noise[start:end]     
    y_noise = np.asarray(y_noise)

    y_clean_pw = np.dot(y_clean,y_clean) 
    y_noise_pw = np.dot(y_noise,y_noise) 

    scalar = np.sqrt(y_clean_pw/((10.0**(SNR/10.0))*y_noise_pw))
    noise = scalar * y_noise
    y_noisy = y_clean + noise

    if normalize:  # Error: only the last one EMG could be saved, but there are different SNRs.
        norm_scalar = np.max(abs(y_noisy))
        y_noisy = y_noisy/norm_scalar
        
    if return_info is False:
        return y_noisy, clean_rate
    else:
        info = {}
        info['start'] = start
        info['end'] = end
        info['scalar'] = scalar
        return y_noisy, clean_rate, info

if __name__ == '__main__':

    args = get_args()
    if args.dataset_type == 'train':
        #Training data
        exercise = 1
        channel = [2]
        EMG_data_num = 40
        SNR_list = [1, -3, -7, -11,-15] 
        num_of_copy = [6,4]
        #normalize = False
    else: 
    # Testing dataset
        exercise = 2
        channel = [11] #range(9,13)
        EMG_data_num = 40
        SNR_list = [2,-2,-6,-10,-14]
        num_of_copy = [6]   
        
    if args.noise_type == "all":
        noise_paths =  [args.noise_path,args.noise_path]
    else:    
        noise_paths = [args.noise_path+args.noise_type,args.noise_path+args.noise_type]

    test = False if args.dataset_type == 'train' else True
    normalize = False # Output noisy EMG without normalization. Set to False in building testing set.
    sti = True
    noisy_folder = 'noisy'
    loop_break = False # Cross channel dataset only needs to add noise for once

    for ch in channel:

        if loop_break == True:
            break
        if args.cross_ch == True:
            out_path = "./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(channel[0])+"_"+str(channel[-1])+args.dir_end
            loop_break = True
        else:
            out_path = "./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(ch)+args.dir_end 
        
        if test == True:
            clean_paths =[out_path+'/test/clean']
        else:
            clean_paths = [out_path+'/train/clean',out_path+'/val/clean']

        print(out_path)
        
        for i in range(len(clean_paths)):
            clean_path = clean_paths[i]
            noise_path = noise_paths[i]
            Noisy_path = clean_path.replace('clean',noisy_folder)
            root_path = clean_path.replace('clean',noisy_folder)
            
            check_path(Noisy_path)
            
            clean_list = get_filepaths(clean_path) if sti is False else get_filepaths_withSTI(clean_path)
            
            noise_list = get_filepaths(noise_path)

            sorted(clean_list)

            for snr in SNR_list:
                    with open(root_path+str(snr)+args.noise_type+'.csv', 'w', newline='') as csvFile:
                        fieldNames = ['EMG',args.noise_type,'start','end','scalar']
                        writer = csv.DictWriter(csvFile, fieldNames)
                        writer.writeheader()
                        for clean_emg_path in tqdm(clean_list):
                                noise_path_list = random.sample(noise_list, num_of_copy[i])
                                for noise_path in noise_path_list:
                                    y_noisy, clean_rate, info = add_noise(clean_emg_path, noise_path, snr, True, normalize)
                                                                            
                                    noise_name = noise_path.split(os.sep)[-1].split(".")[0]
                                    output_dir = Noisy_path+os.sep+str(snr)+os.sep+noise_name
                                    creat_dir(output_dir)
                                    emg_name = clean_emg_path.split(os.sep)[-1].split(".")[0]
                                    np.save(os.path.join(output_dir,emg_name),y_noisy)
                                    writer.writerow({'EMG':emg_name,args.noise_type:noise_name, 'start':info['start'], 'end':info['end'],'scalar':info['scalar']})
      


