import argparse, os, torch, numpy as np
from tqdm import tqdm
from util import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_path', type=str, default='data_E2_S40_Ch11_withSTI_seg2s_F1/train/noisy')
    parser.add_argument('--clean_path', type=str,default='data_E2_S40_Ch11_withSTI_seg2s_F1/train/clean')
    parser.add_argument('--out_path', type=str, default='./trainpt_E1_S40_Ch2_withSTI_seg2s_F1/')
    parser.add_argument('--only_clean', action='store_true', default=False)
    parser.add_argument('--val', action='store_true', default=True)
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--frame_size', type=int, default=2000)
    parser.add_argument('--noise_type', type=str, default='all') # Options: E,P,B,P+B+E
    args = parser.parse_args()
    return args

def get_specific_filepaths(directory,n_type,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        if len(directories) == 0 and n_type in root.split('/')[-1]:
            for filename in files:
                if filename.endswith(ftype):
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)
    return sorted(file_paths)


if __name__ == '__main__':
    args = get_args()
    train_path = args.noisy_path
    clean_path = args.clean_path
    out_path = args.out_path    
    n_frame = args.frame_size
    
    # Generate clean training set
    clean_files = get_filepaths(clean_path)
    if args.val:
        clean_files = clean_files+get_filepaths(clean_path.replace('train','val'))
    # Iterate through all file names in filepath "clean_files"
    cout_path = os.path.join(out_path,'clean')
    for emg_file in tqdm(clean_files):
        emg_name = emg_file.split('/')[-1]
        c_emg = np.load(emg_file)
        c_emg = torch.from_numpy(c_emg)

        for c in range(args.augment):
            for i in np.arange(int(c_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                # Save each segment of data(emma+spec) with n_frame by name folder/emg_name_i.pt
                cout_name = os.path.join(cout_path,emg_name.split(".")[0]+'_'+str(i)+'_'+str(c)+'.pt')
                # Create a folder with cout_path if not exist
                check_folder(cout_name)
                # Save emg data by n_frame
                torch.save( c_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone() ,cout_name)

    # Generate noisy training file
    noisy_files = get_filepaths(train_path) if args.noise_type == 'all' else get_specific_filepaths(train_path,args.noise_type)
    for emg_file in tqdm(noisy_files):
        if args.only_clean:
            break
        emg_name = emg_file.split('/')[-1]
        noise = emg_file.split(os.sep)[-2]
        snr = emg_file.split(os.sep)[-3]
        nout_path = os.path.join(out_path,'noisy',snr,noise,emg_name.split(".")[0])
        n_emg = np.load(emg_file)
        n_emg = torch.from_numpy(n_emg)

        for c in range(args.augment):
            for i in np.arange(int(n_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                nout_name = nout_path+'_'+str(i)+'_'+str(c)+'.pt'
                check_folder(nout_name)
                torch.save(n_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone() ,nout_name)
    
    if args.val:
        noisy_files = get_filepaths(train_path.replace('train','val'))
        for emg_file in tqdm(noisy_files):
            if args.only_clean:
                break
            emg_name = emg_file.split('/')[-1]
            noise = emg_file.split(os.sep)[-2]
            snr = emg_file.split(os.sep)[-3]
            nout_path = os.path.join(out_path,'val',snr,noise,emg_name.split(".")[0])
            n_emg = np.load(emg_file)
            n_emg = torch.from_numpy(n_emg)

            for c in range(args.augment): # Data Augmentation
                for i in np.arange(int(n_emg.shape[0]-c*n_frame//args.augment)//n_frame):
                    nout_name = nout_path+'_'+str(i)+'_'+str(c)+'.pt'
                    check_folder(nout_name)
                    torch.save(n_emg[int(c*n_frame//args.augment+i*n_frame):int(c*n_frame//args.augment+(i+1)*n_frame)].clone(),nout_name)
    
    