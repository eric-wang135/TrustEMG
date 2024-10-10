train_dir="sEMG_noise_train"
test_dir="sEMG_noise_test"
test_subject_idx=$1
test_subject_idx_step=$2
dir_tail=$3

rm -r ./data_E1_S40_Ch2_withSTI_seg2s_$dir_tail
rm -r ./trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail

# Clean sEMG data
python3 ./preprocess/clean_emg.py --data_type train --segment_size 2 --threshold 0.01 --folder_tail $dir_tail --test_subject_init_idx $1 --test_subject_idx_step $2
python3 ./preprocess/clean_emg.py --data_type test --segment_size 2 --threshold 0.01 --folder_tail $dir_tail --test_subject_init_idx $1 --test_subject_idx_step $2

# Mixture
python3 ./preprocess/Mixture.py --noise_path ../$train_dir/ --noise_type all --dataset_type train --dir_end _withSTI_seg2s_$dir_tail;
python3 ./preprocess/Mixture.py --noise_path ../$test_dir/ --noise_type all --dataset_type test --dir_end _withSTI_seg2s_$dir_tail;

# Generate training pt files
python3 ./gen_pt_aug.py --noisy_path data_E1_S40_Ch2_withSTI_seg2s_$dir_tail/train/noisy --clean_path data_E1_S40_Ch2_withSTI_seg2s_$dir_tail/train/clean --out_path ./trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --frame_size 2000

