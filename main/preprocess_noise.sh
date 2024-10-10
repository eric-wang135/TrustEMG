echo "Preprocessing noise data..."
# Create sEMG noise dataset
rm -r ../sEMG_noise_train; rm -r ../sEMG_noise_test
rm -r ../ECG_train; rm -r ../ECG_test

mkdir ../sEMG_noise_train ../sEMG_noise_test

    #ECG
python3 preprocess/clean_ECG.py
mv ../ECG_train ../sEMG_noise_train/ECG
mv ../ECG_test ../sEMG_noise_test/ECG
    # Copy ECG samples to balance the data for sampling
cp ../sEMG_noise_test/ECG/E19090.npy ../sEMG_noise_test/ECG/E19090_2.npy
cp ../sEMG_noise_test/ECG/E19140.npy ../sEMG_noise_test/ECG/E19140_2.npy
cp ../sEMG_noise_test/ECG/E19093.npy ../sEMG_noise_test/ECG/E19093_2.npy
cp ../sEMG_noise_test/ECG/E19830.npy ../sEMG_noise_test/ECG/E19830_2.npy


    #MOA
cp ../moa_train ../sEMG_noise_train/MOA -r; cp ../moa_test ../sEMG_noise_test/MOA -r
python3 ./preprocess/noise_creater.py --type MOA --path ../sEMG_noise_train
python3 ./preprocess/noise_creater.py --type MOA --path ../sEMG_noise_test
python3 ./preprocess/noise_creater.py --type QM --path ../sEMG_noise_train --number 6
python3 ./preprocess/noise_creater.py --type QM --path ../sEMG_noise_test --number 4
mv ../sEMG_noise_test/QM/* ../sEMG_noise_test/MOA; rmdir ../sEMG_noise_test/QM
mv ../sEMG_noise_train/QM/* ../sEMG_noise_train/MOA; rmdir ../sEMG_noise_train/QM

    #BW, EM, PLI, WGN
python3 ./preprocess/noise_creater.py --type BW --path ../sEMG_noise_train --number 14
python3 ./preprocess/noise_creater.py --type PLI --path ../sEMG_noise_train  --number 14
python3 ./preprocess/noise_creater.py --type WGN --path ../sEMG_noise_train  --number 14
python3 ./preprocess/noise_creater.py --type BW --path ../sEMG_noise_test --number 8
python3 ./preprocess/noise_creater.py --type PLI --path ../sEMG_noise_test --number 8
python3 ./preprocess/noise_creater.py --type WGN --path ../sEMG_noise_test --number 8

    #Compound noise
python3 ./preprocess/noise_creater.py --type PLI_BW_ECG_MOA_WGN --mix_num 3 --path ../sEMG_noise_train --number 7
python3 ./preprocess/noise_creater.py --type PLI_BW_ECG_MOA_WGN --mix_num 3 --path ../sEMG_noise_test --number 4

python3 ./preprocess/noise_creater.py --type PLI_BW_ECG_MOA_WGN --mix_num 5 --path ../sEMG_noise_train --number 70
python3 ./preprocess/noise_creater.py --type PLI_BW_ECG_MOA_WGN --mix_num 5 --path ../sEMG_noise_test --number 40