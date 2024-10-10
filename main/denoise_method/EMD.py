from util import *
from emd import sift
from PyEMD import CEEMDAN
from scipy import signal as sig
import pywt
from skimage.restoration import denoise_wavelet
from scipy.stats.stats import pearsonr

MAX_IMF_NUM = 8

# Calculate noise indices
"""
WGN_NOISE_IDXS = np.zeros(MAX_IMF_NUM)
for i in range(5):
  wgn = 2**i*np.random.normal(0, 1, 2000)
  imf_opts = {'sd_thresh': 0.01}
  wgn_imf = emd.sift.ensemble_sift(wgn, max_imfs=MAX_IMF_NUM, nensembles=40, nprocesses=6, ensemble_noise=1, imf_opts=imf_opts) #[N, K]   
  noise_idx = np.std(wgn_imf,axis=0)
  noise_idx = (noise_idx / noise_idx[0])
  #print(noise_idx)
  WGN_NOISE_IDXS = np.add(WGN_NOISE_IDXS, noise_idx)
WGN_NOISE_IDXS = WGN_NOISE_IDXS/5
"""

# Calculated noise indices
WGN_NOISE_IDXS = np.array([1, 0.4751015, 0.29994139, 0.21576524, 0.13714082, 0.0926146, 0.0560073, 0.05497522])
WGN_NOISE_IDXS_CEEMD = np.array([1.0, 0.45871, 0.309190, 0.19652, 0.14093, 0.084863, 0.06222, 0.05597])


def EMD_method(n_emg,emg_sti,noise_type):
    # This function applies EMD and wavelet threshold to remove noise
    N = n_emg.shape[-1]
    
    b_l_10, a_l_10 = signal.butter(4, 10, 'lp',fs=1000) # remove baseline wander
    b_l_20, a_l_20 = signal.butter(4, 20, 'lp',fs=1000) # remove EM
    b_lp40, a_lp40 = sig.butter(4, 40, 'lp',fs=1000) 
    b_h_30, a_h_30 = sig.butter(4, 30, 'hp',fs=1000)
    b_h_40, a_h_40 = signal.butter(4, 40, 'hp',fs=1000)
    freq = np.fft.fftshift(np.fft.fftfreq(N,0.001))
    
    imf_opts = {'sd_thresh': 0.01}
    # Apply EMD to get IMFs
    u = sift.sift(n_emg, max_imfs=MAX_IMF_NUM, imf_opts=imf_opts) #[N, K]
    u_spectrum = abs(np.fft.fft(u,axis=0))

    for i in range(u.shape[-1]): 
        
        P_max = freq[N//2+np.argmax(u_spectrum[:N//2,i])]
        
        if 'P' in noise_type and 50<P_max<70:
            b_n,a_n = sig.iirnotch(P_max,20,fs=1000)
            u[:,i] = sig.filtfilt(b_n,a_n,u[:,i])
        
        if 'm' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = sig.filtfilt(b_h_40,a_h_40,u[:,i])

        elif 'Q' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] =  u[:,i] - sig.filtfilt(b_l_20,a_l_20,u[:,i])

        elif 'B' in noise_type:
            if P_max < 10:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l_10,a_l_10,u[:,i])

    if 'WG' in noise_type:
        # EMD noise indices thresholding
        imf_std = np.std(u,axis=0)
        estimated_wgn_std = imf_std[0] * WGN_NOISE_IDXS
        u[:,0] = 0 # First IMF is usually noise
        for i in range(1, u.shape[-1]):
            if estimated_wgn_std[i]>imf_std[i] or imf_std[0] == 0:
                # IMF contains noise only 
                u[:,i] = 0
            else:
                # IMF is a noisy sEMG signal
                u[:,i] = denoise_wavelet(u[:,i], sigma = estimated_wgn_std[i], method='VisuShrink', mode='soft', wavelet_levels=6, wavelet='sym8', rescale_sigma='True')
            
    if 'E' in noise_type:
        
        if 'B' in noise_type or 'm' in noise_type or 'Q' in noise_type:
            n_emg =  n_emg - sig.filtfilt(b_l_10,a_l_10,n_emg)
        
        u = u[:,~np.all(u == 0, axis=0)]
        ecg_template =  sig.filtfilt(b_lp40, a_lp40, n_emg) # ECG template
        
        corr = []
        for i in range(u.shape[-1]):
            corr.append(pearsonr(ecg_template,u[:,i])[0])

        # Remove/highpass filter top-3 IMFs highly correlated with ECG template
        ecg_components = np.argsort(corr, axis=0)[-6:]
        for i in range(ecg_components.shape[0]):
            u[:,ecg_components[i]] = sig.filtfilt(b_h_30,a_h_30,u[:,ecg_components[i]])
        
        enh_emg = np.sum(u[:,:4],axis=1)
        enh_emg = sig.filtfilt(b_h_30,a_h_30,enh_emg)

    else:
        enh_emg = np.sum(u,axis=1)

    return enh_emg

def EEMD_method(n_emg,emg_sti,noise_type):    
    # This function applies EEMD and wavelet threshold to remove noise
    N = n_emg.shape[-1]
    
    b_l, a_l = signal.butter(4, 10, 'lp',fs=1000) # Good for remove baseline wander
    b_l_20, a_l_20 = signal.butter(4, 20, 'lp',fs=1000) # Good for remove EM
    b_lp40, a_lp40 = sig.butter(4, 40, 'lp',fs=1000) 
    b_h_30, a_h_30 = signal.butter(4, 30, 'hp',fs=1000)
    b_h_40, a_h_40 = signal.butter(4, 40, 'hp',fs=1000)
    freq = np.fft.fftshift(np.fft.fftfreq(N,0.001))
    
    imf_opts = {'sd_thresh': 0.01}
    # Apply EEMD to get IMFs
    u = emd.sift.ensemble_sift(n_emg, max_imfs=MAX_IMF_NUM, nensembles=100, nprocesses=6, ensemble_noise=0.2*np.max(abs(n_emg)), imf_opts=imf_opts)
    u_spectrum = abs(np.fft.fft(u,axis=0))

    for i in range(u.shape[-1]): 
        
        P_max = freq[N//2+np.argmax(u_spectrum[:N//2,i])]
        
        if 'P' in noise_type and 50<P_max<70:
            b_n,a_n = sig.iirnotch(P_max,20,fs=1000)
            u[:,i] = sig.filtfilt(b_n,a_n,u[:,i])

        if 'm' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = sig.filtfilt(b_h_40,a_h_40,u[:,i])

        elif 'Q' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] =  u[:,i] - sig.filtfilt(b_l_20,a_l_20,u[:,i])

        elif 'B' in noise_type:
            if P_max < 10:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l,a_l,u[:,i])

    if 'WG' in noise_type:
    
        # EEMD noise indices thresholding
        imf_std = np.std(u,axis=0)
        estimated_wgn_std = imf_std[0] * WGN_NOISE_IDXS
        u[:,0] = 0 #First IMF is usually noise
        for i in range(1, u.shape[-1]):
            if estimated_wgn_std[i]>imf_std[i] or imf_std[0] == 0: 
                # IMF contains noise only 
                u[:,i] = 0
            else:
                # IMF is a noisy sEMG signal
                u[:,i] = denoise_wavelet(u[:,i], sigma = estimated_wgn_std[i], method='VisuShrink', mode='soft', wavelet_levels=6, wavelet='sym8', rescale_sigma='True')
            #u[:,i] = wavelet_threshold(u[:,i],T[i])    

    if 'E' in noise_type:
        if 'B' in noise_type or 'm' in noise_type or 'Q' in noise_type:
            n_emg =  n_emg - sig.filtfilt(b_l,a_l,n_emg)

        # Get ECG template
        ecg_template =  sig.filtfilt(b_lp40, a_lp40, n_emg) 
        u = u[:,~np.all(u == 0, axis=0)]

        corr = []
        for i in range(u.shape[-1]):
            corr.append(pearsonr(ecg_template,u[:,i])[0])
            
        # Remove/highpass filter top-3 IMFs highly correlated with ECG template
        ecg_components = np.argsort(corr, axis=0)[-6:]
        for i in range(ecg_components.shape[0]):
            u[:,ecg_components[i]] = sig.filtfilt(b_h_30,a_h_30,u[:,ecg_components[i]])
        
        enh_emg = np.sum(u[:,:4],axis=1)
        enh_emg = sig.filtfilt(b_h_30,a_h_30,enh_emg)

    else:
        enh_emg = np.sum(u,axis=1)

    return enh_emg

def CEEMDAN_method(n_emg,emg_sti,noise_type):
    # This function applies CEEMDAN and wavelet threshold to remove noise
    ceemdan = CEEMDAN(range_thr=0.001, total_power_thr=0.01,trials=20)

    N = n_emg.shape[-1]
    
    b_l, a_l = signal.butter(4, 10, 'lp',fs=1000) # Good for remove baseline wander
    b_l_20, a_l_20 = signal.butter(4, 20, 'lp',fs=1000) # Good for remove EM
    b_lp40, a_lp40 = sig.butter(4, 40, 'lp',fs=1000) 
    b_h_30, a_h_30 = signal.butter(4, 30, 'hp',fs=1000)
    b_h_40, a_h_40 = signal.butter(4, 40, 'hp',fs=1000)
    freq = np.fft.fftshift(np.fft.fftfreq(N,0.001))
    
    # Apply CEEMDAN to get IMFs
    u = ceemdan(n_emg,max_imf=MAX_IMF_NUM)#[:-1,:]
    u = u.T # u shape = [imf_num,time]
    u_spectrum = abs(np.fft.fft(u,axis=0))

    for i in range(u.shape[-1]): 
        
        P_max = freq[N//2+np.argmax(u_spectrum[:N//2,i])]
        
        if 'P' in noise_type and 50<P_max<70:
            b_n,a_n = sig.iirnotch(P_max,20,fs=1000)
            u[:,i] = sig.filtfilt(b_n,a_n,u[:,i])

        if 'm' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = sig.filtfilt(b_h_40,a_h_40,u[:,i])

        elif 'Q' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] =  u[:,i] - sig.filtfilt(b_l_20,a_l_20,u[:,i])

        elif 'B' in noise_type:
            if P_max < 10:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l,a_l,u[:,i])

    if 'WG' in noise_type:
    
        # CEEMDAN noise indices thresholding
        imf_std = np.std(u,axis=0)
        estimated_wgn_std = imf_std[0] * WGN_NOISE_IDXS_CEEMD
        u[:,0] = 0 # The first IMF is usually noise
        for i in range(1, u.shape[-1]-1):
            if estimated_wgn_std[i]>imf_std[i] or imf_std[0] == 0:
                # IMF contains noise only 
                u[:,i] = 0
            else:
                # IMF is a noisy sEMG signal
                u[:,i] = denoise_wavelet(u[:,i], sigma = estimated_wgn_std[i], method='VisuShrink', mode='soft', wavelet_levels=6, wavelet='sym8', rescale_sigma='True')

    if 'E' in noise_type:
        if 'B' in noise_type or 'm' in noise_type or 'Q' in noise_type:
            n_emg =  n_emg - sig.filtfilt(b_l,a_l,n_emg)

        ecg_template =  sig.filtfilt(b_lp40, a_lp40, n_emg) # ECG template
        u = u[:,~np.all(u == 0, axis=0)]

        corr = []
        for i in range(u.shape[-1]):
            corr.append(pearsonr(ecg_template,u[:,i])[0])

        # Remove/highpass filter top-3 IMFs highly correlated with ECG template
        ecg_components = np.argsort(corr, axis=0)[-6:]
        for i in range(ecg_components.shape[0]):
            u[:,ecg_components[i]] = sig.filtfilt(b_h_30,a_h_30,u[:,ecg_components[i]])
        
        enh_emg = np.sum(u[:,:4],axis=1)
        enh_emg = sig.filtfilt(b_h_30,a_h_30,enh_emg)

    else:
        enh_emg = np.sum(u,axis=1)

    return enh_emg
