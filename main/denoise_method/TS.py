from util import *

def fun(scalar,waveform,template):
  mse =  sum((waveform - scalar * template) ** 2)
  return mse

def get_scalar(waveform,template):
  scalar_0 = 1.0
  wt = (waveform,template)
  result = spo.minimize(fun,scalar_0,wt)
  if result.success:
    return result.x
  else: 
    return 1


def filtered_template_subtraction(n_emg,fc=50):
    error = 0
    fs = 1000
    pad = 500
    b, a = signal.butter(4, [2.5,fc], 'bp',fs=fs) # For peak-detection
    clean_signal = np.pad(signal.filtfilt(b,a,n_emg),(pad,pad))

    # Peak detection
    signal_rec = abs(clean_signal)
    movingavg_1 = np.ones(fs*1)
    movingavg_2 = np.ones(int(fs*0.1))
    signal_1 = np.convolve(signal_rec,movingavg_1,'same')/1000 #1s ma
    signal_2 = np.convolve(signal_rec,movingavg_2,'same')/100 #0.1s ma

    r_peaks = []
    j, mark = 0, 0

    for i in range(clean_signal.shape[0]):
        if i < mark:
            continue
        if signal_1[i]<signal_2[i]:
            for j in range(i,clean_signal.shape[0]):
                if signal_1[j]>signal_2[j]:
                    mark = j
                    #if j-i < 180: # Too close, may be noise
                    if j-i < 140:
                        break
                    peak_idx = i+np.where(clean_signal[i:j] == np.amax(clean_signal[i:j]))[0][0]
                    r_peaks.append(peak_idx)
                    break
    
    if len(r_peaks) == 0:
        # No ECG detected
        return n_emg, error

    # Get template
    if r_peaks[0]<pad:
        # first ECG start time is detected in padding region
        r_peaks.pop(0)

    waveform = []
    template = []
    peak_number = len(r_peaks)
    
    if len(r_peaks) == 1:
        # No ECG detected
        left = 100
        right = 180
    else:
        trr = min([j-i for i, j in zip(r_peaks[:-1], r_peaks[1:])]) # Get the minimum R-R interval
        left = math.floor(0.25*trr)
        right = math.floor(0.45*trr)
        
        for i in range(peak_number-1):
            waveform.append(clean_signal[r_peaks[i]-left:r_peaks[i]+right+1])

    waveform.append(clean_signal[r_peaks[-1]-left:r_peaks[-1]+right+1])
    
    template = waveform

    # Concatenate template and alignment
    clean_signal = clean_signal[pad:-pad]
    r_peaks = [p-pad for p in r_peaks]

    all_template = template[0]
    
    if peak_number > 1:   
        for i in range(1,peak_number):
            all_template = np.concatenate((all_template,np.zeros(r_peaks[i]-r_peaks[i-1]-left-right-1),template[i]))

    l_pad = 0 if r_peaks[0]-left < 0 else r_peaks[0]-left

    if clean_signal.shape[0]-all_template.shape[0]-r_peaks[0]+left+1 < 0: 
        all_template = np.pad(all_template,(l_pad,0)) 
    else: 
        all_template = np.pad(all_template,(l_pad,clean_signal.shape[0]-all_template.shape[0]-r_peaks[0]+left+1))

    if l_pad == 0:
        all_template = all_template[-(r_peaks[0]-left):]
    
    # Subtraction
    enh_EMG = n_emg - all_template[:n_emg.shape[0]] 
    return enh_EMG, error

