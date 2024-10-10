import os, argparse, numpy as np
import scipy.io
from scipy import signal
from util import *
from tqdm import tqdm

def get_emg_filepaths(directory,number,exercise):
    # import n(umber) of EMG signals
    emg_paths =[]
    for i in range(1,number+1):
        filename = "DB2_s"+str(i)+"/S"+str(i)+"_E"+str(exercise)+"_A1.mat"
        emg_paths.append(os.path.join(directory, filename))
    return emg_paths

def read_emg(emg_path,channel,restimulus=True,normalize=True):
    # extract nth channel EMG, bandpass,  down-sampling, normalize
    b, a = signal.butter(4, [20,500], 'bp',fs=2000) #bandpass = signal.butter(4, [20,500], 'bandpass',output='sos',fs=2000)
    emg_data = scipy.io.loadmat(emg_path)
    y_clean = emg_data.get('emg')[:,channel-1] #channel 
    y_clean = signal.filtfilt(b,a,y_clean)[::2] #y_clean = signal.sosfilt(bandpass,y_clean)[::2] 
    if normalize:
        y_clean = y_clean/np.max(abs(y_clean))
        #print('norm')
    y_clean = y_clean.astype('float64').squeeze()

    y_restimulus = emg_data.get('restimulus')[::2].squeeze() #if restimulus else 0

    return y_clean, y_restimulus

def is_triggered(c_data,sti_data,threshold=0.02):

    for i,amplitude in enumerate(sti_data):
            if amplitude>0:
                sti_data[i] = 1

    # Segment not triggered if 90% of the segment is Inactive State (No action is performed) 
    AS_ratio = np.sum(sti_data)/len(sti_data)
    #print(AS_ratio)
    #input('?')
    if AS_ratio < 0.1:
        #print("Not triggered")
        return False
    elif AS_ratio == 1:
        return True
    
    sti_emg = np.multiply(c_data,sti_data)
    rest_emg = c_data-sti_emg
    sti_emg,rest_emg = sti_emg[sti_emg!=0],rest_emg[rest_emg!=0]
    std_sti, std_rest = np.std(sti_emg),np.std(rest_emg)
    
    if std_sti-std_rest < threshold:
        return False
    
    return True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='train', help='train or test')
    parser.add_argument('--segment_size', type=int, default=2, help='sEMG segment size (second)')
    parser.add_argument('--cross_ch',  action='store_true', default=False, help='Cross channel dataset')
    parser.add_argument('--remove_weak', action='store_true', default=False, help='Remove weak sEMG segments')
    parser.add_argument('--threshold', type=float, default=0.02, help='standard deviation of sEMG segments')
    parser.add_argument('--folder_tail', type=str, default='', help='tail of the output folder name')
    parser.add_argument('--test_subject_init_idx', type=int, default='', help='the starting number of subject for the test set')
    parser.add_argument('--test_subject_idx_step', type=int, default=1, help='the interval number of subject for the test set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    data_type = args.data_type

    if data_type == 'train':
        exercise = 1
        channel = [2]
    else:
        exercise = 2
        channel = [11]#[9,10,11,12]

    Corpus_path = '../EMG_DB2/'
    EMG_data_num = 40
    cross_ch = args.cross_ch #True
    segment = args.segment_size # unit: second
    points_per_seg = segment * 1000 # fs = 1000 Hz
    total_count = 0
    test_subject_idx_lists = list(range(args.test_subject_init_idx,args.test_subject_init_idx+10*args.test_subject_idx_step,args.test_subject_idx_step))
    test_subject_idx_lists = [x%40 for x in test_subject_idx_lists]

    print("test set subjects:",test_subject_idx_lists)

    for ch in channel:
        count = 0
        if cross_ch == True:
            out_path ="./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(channel[0])+"_"+str(channel[-1])+"_withSTI_seg"+str(args.segment_size)+"s_"+args.folder_tail
        else:
            out_path ="./data_E"+str(exercise)+"_S"+str(EMG_data_num)+"_Ch"+str(ch)+"_withSTI_seg"+str(args.segment_size)+"s_"+args.folder_tail

        check_path(out_path)

        if data_type == 'train':
            check_path(out_path+'/train/clean')
            check_path(out_path+'/val/clean')
        else:
            check_path(out_path+'/test/clean')

        file_paths = get_emg_filepaths(Corpus_path,EMG_data_num,exercise)
        train_subject_num = 0
        for i in tqdm(range(len(file_paths))):
            test = False   
            if i not in test_subject_idx_lists:
                
                if data_type == 'test':
                    continue

                if train_subject_num > 24:
                    save_path = out_path+'/val/clean'
                    print("subject for validation:",i)
                else :
                    save_path = out_path+'/train/clean'
                    train_subject_num += 1
                    print("subject for train:",i)
            else:
                if data_type == 'train':
                    continue    
                save_path = out_path+'/test/clean'
                test = True

            emg_file,restimulus = read_emg(file_paths[i],ch,test)
            
            for j in range(emg_file.shape[0]//points_per_seg):
                
                if is_triggered(emg_file[j*points_per_seg:(j+1)*points_per_seg],restimulus[j*points_per_seg:(j+1)*points_per_seg],args.threshold):
                    #print("save")
                    np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)),emg_file[j*points_per_seg:(j+1)*points_per_seg])

                    if test:                        
                        np.save(os.path.join(save_path,file_paths[i].split(os.sep)[-1].split(".")[0]+"_ch"+str(ch)+"_"+str(j)+"_sti"),restimulus[j*points_per_seg:(j+1)*points_per_seg])
                
                    count += 1
        
        print(count, " segments of sEMG is generated in channel",out_path)
        total_count += count

    print(count, " segments of sEMG is generated in ",out_path)
            
