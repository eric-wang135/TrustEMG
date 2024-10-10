dir_tail=$1
test_noisy="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/"
test_clean="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/"
gpu=$2
epoch=$3

# Train

python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_RM --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR --mode test
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_DM  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR --mode test
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model CNN_waveform  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR --mode test
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model FCN  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4  --lr_scheduler MultistepLR --mode test
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_UNetonly  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR --mode test
