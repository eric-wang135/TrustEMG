# TrustEMG-Net: Using Representation-Masking Transformer with U-Net for Surface Electromyography Enhancement

The implementation of TrustEMG-Net, a neural-network-based method for Surface electromyography (sEMG) contaminant removal. 

# Introduction
sEMG is a widely employed bio-signal that captures human muscle activity via electrodes placed on the skin. Several studies have proposed methods to remove sEMG contaminants, as non-invasive measurements render sEMG susceptible to various contaminants. However, these approaches often rely on heuristic-based optimization and are sensitive to the contaminant type. A more potent, robust, and generalized sEMG denoising approach should be developed for various healthcare and human-computer interaction applications. This paper proposes a novel neural network (NN)-based sEMG denoising method called TrustEMG-Net. It leverages the potent nonlinear mapping capability and data-driven nature of NNs. TrustEMG-Net adopts a denoising autoencoder structure by combining U-Net with a Transformer encoder using a representation-masking approach. The proposed approach is evaluated using the Ninapro sEMG database with five common contamination types and signal-to-noise ratio (SNR) conditions. Compared with existing sEMG denoising methods, TrustEMG-Net achieves exceptional performance across the five evaluation metrics, exhibiting a minimum improvement of 20%. Its superiority is consistent under various conditions, including SNRs ranging from -14 to 2 dB and five contaminant types. An ablation study further proves that the design of TrustEMG-Net contributes to its optimality, providing high-quality sEMG and serving as an effective, robust, and generalized denoising solution for sEMG applications.

For more detail please check our <a href="https://arxiv.org/abs/2410.03843" target="_blank">Paper</a>

### Setup ###

You can apply our environmental setup in the Environment Folder using the following script.

```js
conda env create -f environment.yml
```

### Database ###

Please download the open-access databases from these websites first:
1. sEMG: [NINAPro database DB2](http://ninaweb.hevs.ch/node/17)
2. ECG artifact: [MIT-BIH Normal Sinus Rhythm Database](https://www.physionet.org/content/nsrdb/1.0.0/) 
3. Baseline wander (BW) and Motion artifact (MOA): [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/)

Please place the downloaded datasets and the main folder as follows:

    ├─- mit-bih-noise-stress-test-database-1.0.0
    ├─- mit-bih-normal-sinus-rhythm-database-1.0.0
    ├─- EMG_DB2
    ├─- moa_train (from a private dataset in "Deep learning for surface electromyography artifact contamination type detection.")
    ├─- moa_test (from a private dataset in "Deep learning for surface electromyography artifact contamination type detection.")
    └─- main

### Preprocess contaminant data

To generate the contamination data, please execute the following script:

``` js
bash preprocess_noise.sh
```

### Training and Inference ###

To generate noisy sEMG data and obtain the results of all the sEMG denoising methods for a four-fold subject-wise cross-validation test, please run the following script:

``` js
bash run_four_folds.sh
```

By default, the script uses sEMG data from subjects 1–10, 11–20, 21–30, and 31–40 for the test set across the four folds.

If you only need the results for one fold, please run the following script:

``` js
bash run_one_fold.sh
```

Note that the CEEMDAN-based and VMD-based methods require a long processing time. 
If the corresponding results are not essential for your analysis, you can modify the relevant scripts in "conventional.sh" to skip these methods.

### References ###

Please kindly cite our paper if you find this code useful.

    @misc{wang2024trustemgnetusingrepresentationmaskingtransformer,
        title={TrustEMG-Net: Using Representation-Masking Transformer with U-Net for Surface Electromyography Enhancement}, 
        author={Kuan-Chen Wang and Kai-Chun Liu and Ping-Cheng Yeh and Sheng-Yu Peng and Yu Tsao},
        year={2024},
        eprint={2410.03843},
        archivePrefix={arXiv},
        primaryClass={eess.SP},
        url={https://arxiv.org/abs/2410.03843}, 
    }

If you use these databases, please cite these papers:
  1. Ninapro database

    @article{atzori2014electromyography,
      title={Electromyography data for non-invasive naturally-controlled robotic hand prostheses},
      author={Atzori, Manfredo and Gijsberts, Arjan and Castellini, Claudio and Caputo, Barbara and Hager, Anne-Gabrielle Mittaz and Elsig, Simone and Giatsidis, Giorgio and Bassetto, Franco and M{\"u}ller, Henning},
      journal={Scientific data},
      volume={1},
      number={1},
      pages={1--13},
      year={2014},
      publisher={Nature Publishing Group}
    }
  2. MIT-BIH NSRD

    @article{goldberger2000physiobank,
      title={PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals},
      author={Goldberger, Ary L and Amaral, Luis AN and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
      journal={circulation},
      volume={101},
      number={23},
      pages={e215--e220},
      year={2000},
      publisher={Am Heart Assoc}
    }
  3. MIT-BIH NSTDB

    @article{moody1984noise,
    title={A noise stress test for arrhythmia detectors},
    author={Moody, George B and Muldrow, WE and Mark, Roger G},
    journal={Computers in cardiology},
    volume={11},
    number={3},
    pages={381--384},
    year={1984}
  }  

### Note ###
1. A supplementary material detailing the implementation of the comparison methods is available in the **doc** folder of this repository.
2. The code implementation of the variational mode decomposition (VMD) utilized in the VMD-based sEMG denoising method is sourced from <a href="https://github.com/vrcarva/vmdpy" target="_blank">this repository</a>.
3. In this code, the EM data from the MIT-BIH NSTDB is labeled as "QM" to better distinguish it from other contamination types, such as ECG and MOA.
4. We would like to thank Professor Juliano Machado from the Department of Basic, Technical, and Technological Education at Sul-Rio-Grandense Federal Institute for generously providing the MOA dataset used in this research.
