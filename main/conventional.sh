dir_tail=$1
test_noisy="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/"
test_clean="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/"

python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/   --mode test --task IIR 
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/   --mode test --task FTS+IIR 
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/   --mode test --task EMD
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/   --mode test --task VMD
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/   --mode test --task CEEMDAN
