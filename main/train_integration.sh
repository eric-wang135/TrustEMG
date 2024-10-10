dir_tail=$1
test_noisy="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/"
test_clean="./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/"
gpu=$2
epoch=$3

#trained_epochs=$3 #1000


# Train lstm Integration methods

python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_LSTM_DM  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_LSTM_RM  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_skipall_DM --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR
python3 main.py --train_path trainpt_E1_S40_Ch2_withSTI_seg2s_$dir_tail/ --test_noisy ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/noisy/ --test_clean ./data_E2_S40_Ch11_withSTI_seg2s_$dir_tail/test/clean/  --model TrustEMGNet_skipall_RM  --epochs $epoch --lr 0.001 --batch_size 256 --loss_fn l1 --gpu $gpu --worker_num 4 --lr_scheduler MultistepLR

