python3 ./preprocess/Mixture.py --noise_path ../sEMG_noise_test/ --noise_type all --dataset_type test --dir_end _withSTI_seg2s_05
python3 ./gen_pt_aug.py --noisy_path data_E1_S40_Ch2_withSTI_seg2s_05/train/noisy --clean_path data_E1_S40_Ch2_withSTI_seg2s_05/train/clean --out_path ./trainpt_E1_S40_Ch2_withSTI_seg2s_05/ --frame_size 2000