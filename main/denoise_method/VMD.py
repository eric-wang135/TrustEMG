import numpy as np
from scipy import signal as sig


def VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    
    
    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
                       1 = all omegas start uniformly distributed
                      2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6
    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """
    
    if len(f)%2:
       f = f[:-1]

    # Period and sampling frequency of input signal
    
    ltemp = len(f)//2 
    fMirr =  np.append(np.flip(f[:ltemp],axis = 0),f)  
    fMirr = np.append(fMirr,np.flip(f[-ltemp:],axis = 0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1,T+1)/T  
    
    # Spectral Domain discretization
    freqs = t-0.5-(1/T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha*np.ones(K)
    
    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat) #copy f_hat
    f_hat_plus[:T//2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])


    if init == 1:
        for i in range(K):
            omega_plus[0,i] = (0.5/K)*(i)
    elif init == 2:
        fs = 1./len(f)
        omega_plus[0,:] = np.sort(np.exp(np.log(fs) + (np.log(0.5)-np.log(fs))*np.random.rand(1,K)))
    else:
        omega_plus[0,:] = 0
            
    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0,0] = 0
    
    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype = complex)
    
    # other inits
    uDiff = tol+np.spacing(1) # update step
    n = 0 # loop counter
    sum_uk = 0 # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K],dtype=complex)    

    #*** Main loop for iterative updates***

    while ( uDiff > tol and  n < Niter-1 ): # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n,:,K-1] + sum_uk - u_hat_plus[n,:,0]
        
        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1.+Alpha[k]*(freqs - omega_plus[n,k])**2)
        
        # update first omega if not held at 0
        if not(DC):
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)

        # update of any other mode
        for k in np.arange(1,K):
            #accumulator
            sum_uk = u_hat_plus[n+1,:,k-1] + sum_uk - u_hat_plus[n,:,k]
            # mode spectrum
            u_hat_plus[n+1,:,k] = (f_hat_plus - sum_uk - lambda_hat[n,:]/2)/(1+Alpha[k]*(freqs - omega_plus[n,k])**2)
            # center frequencies
            omega_plus[n+1,k] = np.dot(freqs[T//2:T],(abs(u_hat_plus[n+1, T//2:T, k])**2))/np.sum(abs(u_hat_plus[n+1,T//2:T,k])**2)
            
        # Dual ascent
        lambda_hat[n+1,:] = lambda_hat[n,:] + tau*(np.sum(u_hat_plus[n+1,:,:],axis = 1) - f_hat_plus)
        
        # loop counter
        n = n+1
        
        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1/T)*np.dot((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i]),np.conj((u_hat_plus[n,:,i]-u_hat_plus[n-1,:,i])))

        uDiff = np.abs(uDiff)        
    
    #discard empty space if converged early
    Niter = np.min([Niter,n])
    omega = omega_plus[:Niter,:]
    
    idxs = np.flip(np.arange(1,T//2+1),axis = 0)
    # Signal reconstruction
    u_hat = np.zeros([T, K],dtype = complex)
    u_hat[T//2:T,:] = u_hat_plus[Niter-1,T//2:T,:]
    u_hat[idxs,:] = np.conj(u_hat_plus[Niter-1,T//2:T,:])
    u_hat[0,:] = np.conj(u_hat[-1,:])    
    
    u = np.zeros([K,len(t)])
    for k in range(K):
        u[k,:] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:,k])))
        
    # remove mirror part
    u = u[:,T//4:3*T//4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1],K],dtype = complex)
    for k in range(K):
        u_hat[:,k]=np.fft.fftshift(np.fft.fft(u[k,:]))

    return u, u_hat, omega


def VMD_denoise(n_emg,emg_sti,noise_type):
    segment_length = n_emg.shape[-1]

    b_l, a_l = sig.butter(4, 10, 'lp',fs=1000) # Good for remove baseline wander
    b_l_20, a_l_20 = sig.butter(4, 20, 'lp',fs=1000) # Good for remove baseline wander
    b_h_40, a_h_40 = sig.butter(4, 40, 'hp',fs=1000)
    
    freq = np.fft.fftshift(np.fft.fftfreq(segment_length,0.001))

    #. some sample parameters for VMD  https://zhuanlan.zhihu.com/p/29571585
    alpha = 1000       # moderate bandwidth constraint
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 10            # numebr of IMFs
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7

    u, u_hat, omega = VMD(n_emg, alpha, tau, K, DC, init, tol)  
    u = u.T

    if 'WG' in noise_type:

        noise_sec = np.where( emg_sti == 0)[0]
        signal_sec = np.where( emg_sti > 0)[0]

        if noise_sec.shape[0]>10000 and signal_sec.shape[0]>10000:
            noise = n_emg[noise_sec[:10000]]
            #signal = n_emg[signal_sec[:10000]]-n_emg[noise_sec[:10000]]

        elif noise_sec.shape[0]>20:
            length = noise_sec.shape[0] if noise_sec.shape[0]>signal_sec.shape[0] else signal_sec.shape[0]
            noise = n_emg[noise_sec[:length]]
            #signal = n_emg[signal_sec[:length]]-noise
        else:
            min_std, max_std = 10, -1
            min_std_start, max_std_start = 0, 0
            length = segment_length//10
            for i in range(10):
                segment_std = np.std(n_emg[i*length:(i+1)*length])
                """
                if max_std < segment_std:
                    max_std = segment_std
                    max_std_start = i
                """
                if min_std > segment_std:
                    min_std = segment_std
                    min_std_start = i

            noise = n_emg[min_std_start*length:(min_std_start+1)*length]

        noise_std = np.std(noise)
        wgn = noise_std* np.random.normal(0, 1, segment_length)

        wgn_IMF, u_hat, omega = VMD(wgn, alpha, tau, K, DC, init, tol)

        T = np.sum(np.std(wgn_IMF,1))/K
        C = 1.25

    for i in range(K): 
        
        P_max = freq[segment_length//2+np.argmax(abs(u_hat[segment_length//2:,i]))]
        
        if 'P' in noise_type and 50<P_max<70:
            b_n,a_n = sig.iirnotch(P_max,20,fs=1000)
            u[:,i] = sig.filtfilt(b_n,a_n,u[:,i])
            
        if 'WG' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else: 
                for j in range(segment_length):
                    u[j,i] = 0 if np.abs(u[j,i]) < T else C*u[j,i]*(np.abs(u[j,i])-T)/np.abs(u[j,i])

        if 'E' in noise_type:
            continue
        
        if 'm' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = sig.filtfilt(b_h_40,a_h_40,u[:,i])

        elif 'Q' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l_20,a_l_20,u[:,i])

        elif 'B' in noise_type:
            if P_max < 10:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l,a_l,u[:,i])

    enh_emg = np.sum(u,1)

    return enh_emg   

def get_intervals(input):
  zero_crossings = []
  zero_crossings.append(0)
  for i in range(1, input.shape[-1]):
        # Check for sign change between consecutive samples
        if (input[i-1] >= 0 and input[i] < 0) or (input[i-1] < 0 and input[i] >= 0):
            # Interpolate to find exact zero crossing index
            zero_crossings.append(i)

  # Insert index 0 if it's not included in the zero-crossing list (To ensure getting the first interval)
  if zero_crossings[0] != 0:
    zero_crossings.insert(0,0)

  return zero_crossings

def interval_thr(input, threshold, mode='soft'):
  C = 1
  # Get the extrema of the input interval
  extrema = np.max(np.abs(input))
  if extrema == 0:
    return input
    
  if mode == 'soft':
    return C*input * np.maximum((extrema - threshold) / extrema, 0)
  if mode == 'hard':
    return input * np.maximum(extrema - threshold, 0)
  if mode == 'SCAD':
    z = 3.7
    if extrema <= 2*threshold:
      return C*input * (np.maximum((extrema - threshold)/extrema, 0))
    if extrema <= z*threshold:
      return C*input * ((z-1)*extrema - z*threshold) / ((z-2)*extrema)
    if extrema > z*threshold:
      return input


def interval_thresholding(input,mode='soft'):
  # assume input's dimension is 1
  N = input.shape[0]
  # Calculate the threshold value
  sigma = np.median(np.abs(input))/0.6745
  threshold = sigma * np.sqrt(2*np.log(N))/4
  # Get the positions of zero-crossings
  zero_crossings = get_intervals(input)
  interval_enhanced_list = []
  # Enhance each interval
  for i in range(len(zero_crossings)-1):

    interval_input = input[zero_crossings[i]:zero_crossings[i+1]]
    interval_enhanced_list.append(interval_thr(interval_input, threshold, mode))

  # Enhance the last interval
  if zero_crossings[-1] == N-1:
    interval_enhanced_list.append(interval_thr(input[-1], threshold, mode))
  else:
    interval_input = input[zero_crossings[-1]:]
    interval_enhanced_list.append(interval_thr(interval_input, threshold, mode))

  output = np.hstack(interval_enhanced_list)

  return output

def shuffle_waveform(waveform):
    """
    Randomly shuffle the sample points of an input waveform to construct a new waveform.

    Parameters:
    waveform (numpy array): Input waveform array.

    Returns:
    numpy array: Shuffled waveform array.
    """
    shuffled_waveform = np.copy(waveform)  # Make a copy to avoid modifying the original waveform
    np.random.shuffle(shuffled_waveform)  # Shuffle the sample points of the copied waveform
    return shuffled_waveform

def VMD_IIT_denoise(n_emg,thr_mode,noise_type):
    segment_length = n_emg.shape[-1]
    b_l, a_l = sig.butter(4, 10, 'lp',fs=1000) # Good for remove baseline wander
    b_l_20, a_l_20 = sig.butter(4, 20, 'lp',fs=1000) # Good for remove baseline wander
    b_h_40, a_h_40 = sig.butter(4, 40, 'hp',fs=1000)

    freq = np.fft.fftshift(np.fft.fftfreq(segment_length,0.001))
    
    # some sample parameters for VMD  https://zhuanlan.zhihu.com/p/29571585
    alpha = 1000       # moderate bandwidth constraint
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)
    K = 12            # numebr of IMFs
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-3

    u, u_hat, omega = VMD(n_emg, alpha, tau, K, DC, init, tol)
    u = u.T  

    for i in range(K):

        P_max = freq[segment_length//2+np.argmax(abs(u_hat[segment_length//2:,i]))]

        if 'P' in noise_type and 50<P_max<70:
            b_n,a_n = sig.iirnotch(P_max,20,fs=1000)
            u[:,i] = sig.filtfilt(b_n,a_n,u[:,i])

        if 'E' in noise_type:
            continue

        if 'm' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = sig.filtfilt(b_h_40,a_h_40,u[:,i])

        elif 'Q' in noise_type:
            if P_max < 20:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l_20,a_l_20,u[:,i])

        elif 'B' in noise_type:
            if P_max < 10:
                u[:,i] = 0
            else:
                u[:,i] = u[:,i] - sig.filtfilt(b_l,a_l,u[:,i])
    
    if 'WG' in noise_type:
      num_iter = 20
      
      n_emg_tmp = n_emg
      final_denoised_emg = np.zeros_like(n_emg)

      for iter in range(num_iter):
        if iter == 0:
          u_tmp = np.copy(u)
        else:
          u_tmp, _, _ = VMD(n_emg_tmp, alpha, tau, K, DC, init, tol)
          u_tmp = u_tmp.T

        for i in range(K):
          u_tmp[:,i] = interval_thresholding(u_tmp[:,i],thr_mode)
          
        denoised_emg = np.sum(u_tmp,axis=1)

        final_denoised_emg += denoised_emg

        # create a new version of noisy waveform
        shuffled_noise = shuffle_waveform(u[:,-1])
        n_emg_tmp = np.sum(u[:,:-1], axis=1) + shuffled_noise

      final_denoised_emg = final_denoised_emg / num_iter
    else:
        final_denoised_emg = np.sum(u,1)

    return final_denoised_emg