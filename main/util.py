import os,sys,math
import torch
import numpy as np
from scipy import signal

#from numpy.core.defchararray import find
#import scipy.optimize as spo

def creat_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_path(path):
    # Check if path directory exists. If not, create a file directory
    if not os.path.isdir(path): 
        os.makedirs(path)
   
def check_folder(path):
    # Check if the folder of path exists
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)


def get_filepaths(directory,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def get_specific_filepaths(directory,n_type,ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        if len(directories) == 0 and n_type in root.split('/')[-1]:
            for filename in files:
                if filename.endswith(ftype):
                    filepath = os.path.join(root, filename)
                    file_paths.append(filepath)  # Add it to the list.
    return sorted(file_paths)

def get_filenames(directory,ftype='.npy'):
    file_names = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                file_names.append(filename)  # Add it to the list.

    return sorted(file_names)

def get_fold_num(s):
    parts = s.split('/')
    for part in parts:
        if 'F' in part:
            return part.split('_')[-1]  # Extract the part after the last underscore
    return None

#------------------------------------------------------------------------

def resample(x, fs, fs_2):
    # x needs to be an 1D numpy array
    return signal.resample(x,int(x.shape[0]/fs * fs_2))

def signal_segmentation(in_sig,frame_size=500,frame_shift=250,dtype='npy'):
    #"Segment signals into overlap frames with overlapping"
    sig_len = in_sig.shape[-1]
    nframes = math.ceil((sig_len - frame_size) / frame_shift + 1)
    if dtype =='npy':
        # nframes = (sig_len // (frame_size - frame_shift))
        out = np.zeros(list(in_sig.shape[:-1]) + [nframes, frame_size])
    else: #'torch,tensor'
        out = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, frame_size), device=in_sig.device)
    start = 0
    end = start + frame_size
    k = 0
    for i in range(nframes):
        if end < sig_len:
            out[..., i, :] = in_sig[..., start:end]
            k += 1
        else:
            tail_size = sig_len - start
            out[..., i, :tail_size] = in_sig[..., start:]

        start += frame_shift
        end = start + frame_size
    return out

def OLA(inputs,frame_shift=250,dtype='npy'):
    # Overlap and add method to inverse the segmentation
    #input = 1 x T (frame_size)x N (frame_num)
    nframes = inputs.shape[-2]
    frame_size = inputs.shape[-1]
    frame_step = frame_shift
    sig_length = (nframes - 1) * frame_step + frame_size

    if dtype=='npy':
        sig = np.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype)
        ones = np.zeros_like(sig)
    else: #"Tensor"
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device, requires_grad=False)
        ones = torch.zeros_like(sig)

    start = 0
    end = start + frame_size
    for i in range(nframes):
        sig[..., start:end] += inputs[..., i, :]
        ones[..., start:end] += 1.
        start += frame_step
        end = start + frame_size
    return sig / ones

#------------------------------------------------------------------------

def cal_SNR(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        noise = enhanced - clean
        noise_pw = np.dot(noise,noise)
        signal_pw = np.dot(clean,clean)
        SNR = 10*np.log10(signal_pw/noise_pw)
    else:
        noise = enhanced - clean
        noise_pw = torch.sum(noise*noise,1)
        signal_pw = torch.sum(clean*clean,1)
        SNR = torch.mean(10*torch.log10(signal_pw/noise_pw)).item()
    return round(SNR,3)

def cal_rmse(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        RMSE = np.sqrt(((enhanced - clean) ** 2).mean())
    else:
        RMSE = torch.sqrt(torch.mean(torch.square(enhanced - clean))).item()
    return round(RMSE,6)

def cal_prd(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
         PRD = np.sqrt(np.sum((enhanced - clean) ** 2) / np.sum(clean ** 2)) * 100
    else:
        PRD = torch.mul(torch.sqrt(torch.div(torch.sum(torch.square(enhanced - clean)),torch.sum(torch.square(clean)))),100).item()
    return round(PRD,3)

def cal_ARV(emg):
  win = 200
  ARV = []
  emg = abs(emg)
  for i in range(0,emg.shape[0],win):
    ARV.append((emg[i:i+win]).mean())
  return np.array(ARV)

def cal_MF(emg,stimulus):
  # 10 - 500Hz mean frequency
  freq = np.fft.fftfreq(n=1024,d=0.001)[:513]
  freq[-1] = freq[-1]*-1
  start = next(i for i,v in enumerate(freq) if v >=10)
  freq = np.expand_dims(freq[start:],1)
  spec = np.abs(signal.stft(emg, fs=1000, window='boxcar',nperseg=200,noverlap=0,nfft=1024,boundary='constant')[-1][start:,:])
  rec_win = signal.get_window('boxcar', 200)
  scale = np.sqrt(1.0 / rec_win.sum()**2)
  spec = np.abs(spec / scale)
  
  weighted_f = np.sum(freq*spec,0)
  spec_column_pow = np.sum(spec,0)
  MF = weighted_f / spec_column_pow
  MF = [MF[i] for i,v in enumerate(stimulus[::200]) if v>0]

  return np.array(MF)

def normalize(x):
  return (x-x.mean())/np.std(x)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

# ------------------------------------------------------------------------
def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
    line = []
    line = f'\rEpoch {epoch}/ {epochs}'
    loss = loss/step
    if step==n_step:
        progress = '='*30
    else :
        n = int(30*step/n_step)
        progress = '='*n + '>' + '.'*(29-n)
    eta = time*(n_step-step)/step
    line += f'[{progress}] - {step}/{n_step} |Time :{int(time)}s |ETA :{int(eta)}s  '
    if step==n_step:
        line += '\n'
    sys.stdout.write(line)
    sys.stdout.flush()

