# Implementation details of comparison sEMG denoising methods

## IIR filters
The selection of IIR filters for sEMG contaminant removal in this work is shown in Table 1. Each IIR filter is applied only if the corresponding contaminant exists in sEMG.

| Contaminant | Filter specification |
|-------------|----------------------|
| BW          | $4^{th}$ order Butterworth high-pass filter, $f_c$ 10 Hz |
| PLI         | Notch filter, $f_c$ 60 Hz and quality factor 5 |
| MOA[1]     | $4^{th}$ order Butterworth high-pass filter, $f_c$ 40 Hz |
| MOA[2]     | $4^{th}$ order Butterworth high-pass filter, $f_c$ 20 Hz |
| ECG         | $4^{th}$ order Butterworth high-pass filter, $f_c$ 40 Hz |
| WGN         | $4^{th}$ order Butterworth band-pass filter, $f_c$ 20 and 500 Hz |

*[1] Machado et al., 2021; [2] Moody and Mark, 1984*

*Note:* $f_c$ *denotes the cutoff frequency.*

## TS+IIR
TS+IIR applies TS and IIR filters for ECG and non-ECG contaminants, respectively. TS consists of three steps: ECG detection, template extraction, and ECG subtraction. For ECG detection, zero-padding is first performed on signal segments for 0.5 seconds at the front and end of the segments. We then calculate two moving averages (1 s and 0.1 s) and identify ECG-containing segments when these moving averages intersect twice within a specified time (more than 0.14 seconds in this work)[3]. Since the sEMG segments are short (2 s), we choose a filtering approach to create ECG templates, using a 4th-order Butterworth high-pass filter with a cutoff frequency of 50 Hz[4]. After subtraction of ECG artifacts, we further apply a 4th-order Butterworth high-pass filter with a cutoff frequency of 40 Hz to obtain the best results[5, 6].

*[3] Junior et al., 2019; [4] Marker et al., 2014; [5] Wang et al., 2023; [6] Drake et al., 2006*

## EMD-based and EEMD-based methods
The EMD-based and EEMD-based methods adopt EMD and EEMD for signal decomposition, respectively. The maximum number of IMFs is set to 8 for both methods. As for EEMD, the noise power parameter, np, is set to 0.2 times the maximum rectified value of each sEMG segment, which is dynamic and can achieve better performance in this work. The ensemble time is 100.

The contaminant removal algorithms for both methods refer to previous research[7, 8, 9, 10]. To discern whether certain contaminant types exist in each mode, we use the Fast Fourier Transform to find the frequency with maximum energy in the spectrum, which is denoted as $f_{max}$. We then apply the following algorithms for different types of contaminants if included[9].

- **BW:** Modes with $f_{max}$ less than 10 Hz are removed. Other modes remove BW by subtracting themselves through a 4th-order Butterworth low-pass filter with a cutoff frequency of 10 Hz.
- **PLI:** For modes with $f_{max}$ between 50 and 70 Hz, we apply a narrow-band notch filter with center frequency $f_{max}$ and quality factor 20.
- **ECG:** We calculate the Pearson correlation coefficients of each mode and ECG template to discern modes with more ECG contamination. The ECG template here is extracted from noisy sEMG by applying a 4th-order Butterworth high-pass filter with a cutoff frequency of 40 Hz. For the six modes with higher correlations with the ECG template, we pass them through a 4th-order Butterworth high-pass filter with a cutoff frequency of 30 Hz.
- **MOA:** Modes with $f_{max}$ less than 20 Hz are removed. Other modes are passed through a 4th-order Butterworth high-pass filter with a cutoff frequency of 20 Hz.
- **WGN:** The first mode is removed. For other modes, it is determined whether to remove them based on their standard deviation. If the standard deviation of the mode exceeds the corresponding threshold value, we apply wavelet thresholding to the mode with sym8 as the mother wavelet[7, 8].

*[7] Zhang et al., 2016; [8] Sun et al., 2020; [9] Ma et al., 2020; [10] Naji et al., 2011*

## VMD-based method
The VMD-based method conducts signal decomposition using VMD with parameters set as follows: the number of IMF is 10, the penalty factor is 1000, and the tolerance is 0.001[9]. The initialization of center frequency adopts a uniform distribution.

We follow [9] and apply the following algorithms for different types of contaminants if included. The usage of $f_{max}$ is identical to the (E)EMD-based method. Notably, we directly apply the same IIR filter to eliminate ECG artifacts, as there is currently no VMD-based method for ECG contamination.

- **BW, PLI, and MOA:** Identical to (E)EMD-based method.
- **WGN:** The sub-band thresholding method is applied, and the parameter refers to [9].
