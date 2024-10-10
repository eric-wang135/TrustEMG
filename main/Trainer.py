import torch
import os, time, numpy as np, pandas as pd
from tqdm import tqdm
from scipy import signal
from util import *
from denoise_method.TS import *
from denoise_method.VMD import *
from denoise_method.EMD import *
    
class Trainer:
    def __init__(self, model=None, epochs=0, epoch=0, best_loss=0, optimizer=None, lr_scheduler=None,
                      criterion=None, device='cuda', loader=0, writer=0, model_path='', score_path='', args=None):
        self.task = args.task
        self.used_time = 0
        self.epoch = epoch
        self.epoch_count = 0
        self.epochs = epochs
        self.best_loss = best_loss
        self.best_loss_snr = -100
        if args.task == 'denoise':
            self.model = model.to(device)
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

            self.device = device
            self.loader = loader
            self.criterion = criterion

            self.train_loss = 0
            self.train_snr = 0
            self.val_loss = 0
            self.val_snr = 0
        
            self.writer = writer
            self.model_path = model_path
            self.train_clean = args.test_clean.replace('test','train')

            self.out_folder = self.model.__class__.__name__
        
        self.score_path = score_path
        self.noise_type = args.noise_type
        self.output = args.output

        if args.task != 'denoise':
            self.fc = 40
            self.highpass_ecg = signal.butter(4,self.fc,'highpass',fs=1000)
            self.notch = signal.iirnotch(60, 5, fs=1000)
            self.highpass_bw = signal.butter(4,10,'highpass',fs=1000)
            self.highpass_wgn = signal.butter(4,20,'highpass',fs=1000)
            self.highpass_moa = signal.butter(4,40,'highpass',fs=1000)

        if args.mode == 'train':
            self.train_step = len(loader['train'])
            self.val_step = len(loader['val'])

        self.args = args
        
    def save_checkpoint(self,):
        if self.lr_scheduler is not None:
            state_dict = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'lr_scheduler':self.lr_scheduler.state_dict()
                }
        else:
            state_dict = {
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                }

        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)

    def print_score(self,test_file):
        
        if self.task == 'denoise':
            self.model.eval()

        n_emg = torch.load(test_file)
        c_file = os.path.join(self.train_clean,test_file.split('/')[-1].replace('.pt','.npy'))
        c_emg = np.load(c_file)
        pred = self.model(n_emg)
        loss = self.criterion(pred, c_emg).item()
        
        pred = pred.cpu().detach().numpy()

    def _train_epoch(self):
        self.train_loss = 0
        self.train_snr = 0
        self.model.train()
        t_start =  time.time()
        step = 0
        self._train_step = getattr(self,f'_train_step_mode_{self.task}')
        
        for data in self.loader['train']:
            step += 1
            self._train_step(data)
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
        
        self.lr_scheduler.step()

        self.train_loss /= len(self.loader['train'])
        self.train_snr /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}')
        print(f'train_SNRimp:{self.train_snr}')
#     @torch.no_grad()
    
    def _val_epoch(self):
        self.val_loss = 0
        self.val_snr = 0
        self.model.eval()
        t_start =  time.time()
        step = 0
        self._val_step = getattr(self,f'_val_step_mode_{self.task}')
        for data in self.loader['val']:
            step += 1
            self._val_step(data)
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='val')
        self.val_loss /= len(self.loader['val'])
        self.val_snr /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
        print(f'val_SNRimp:{self.val_snr}')
        if self.best_loss > self.val_loss:
            self.epoch_count = 0
            print(f"Save model to '{self.model_path}'")
            
            self.save_checkpoint()
            self.best_loss = self.val_loss
            self.best_loss_snr = self.val_snr

    def train(self):
        model_name = self.model.__class__.__name__
        #print("Test validation:")
        #self._val_epoch()
        while self.epoch < self.epochs and self.epoch_count<20:
            self._train_epoch()
            self._val_epoch()
            #self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'train': self.train_loss},self.epoch)
            #self.writer.add_scalars(f'{self.args.task}/{model_name}_{self.args.optim}_{self.args.loss_fn}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            self.epoch_count += 1
        print("best loss:",self.best_loss)
        print("best SNRimp:",self.best_loss_snr)
        #self.writer.close()
    
    
    def write_score(self,test_file,test_path):   
    
        outname  = test_file.replace(f'{test_path}','').replace('/','_')
        c_file = os.path.join(self.args.test_clean,test_file.split('/')[-1])
        clean = np.load(c_file)
        stimulus = np.load(c_file.replace('.npy','_sti.npy')) if test_file.split('/')[-5] == 'test' else 0
        noisy = np.load(test_file)
        noise_type = test_file.split('/')[-2]
        
        error = 0
        t_start =  time.time()

        if self.args.task=='denoise':
            self.model.eval()
            
            
            n_emg = torch.from_numpy(noisy).to(self.device).unsqueeze(0).type(torch.float32)
            c_emg = torch.from_numpy(clean).to(self.device).unsqueeze(0).type(torch.float32)
                            
            if self.args.output_latent:
                pred,feature_map, trans_out = self.model(n_emg)
                pred = pred.squeeze()
                feature_map = feature_map.cpu().detach().numpy()
                trans_out = trans_out.cpu().detach().numpy()
            else:
                
                pred = self.model(n_emg)
                
                if type(pred) is tuple:
                    pred = pred[0].squeeze()
                else:
                    pred = pred.squeeze()
                
            loss = self.criterion(pred.squeeze(),c_emg.squeeze()).item()
            
            pred = pred.cpu().detach().numpy()
            loss = 0
            enhanced = pred
            
        elif self.args.task=='evaluate':
            enhanced = noisy

        elif self.args.task=='IIR':
            enhanced = noisy

            if 'P' in noise_type:
                enhanced = signal.filtfilt(self.notch[0],self.notch[1],enhanced).astype('float64')
            if 'E' in noise_type:
                enhanced = signal.filtfilt(self.highpass_ecg[0],self.highpass_ecg[1],enhanced).astype('float64')
            elif 'm' in noise_type:
                enhanced = signal.filtfilt(self.highpass_moa[0],self.highpass_moa[1],enhanced).astype('float64')
            elif 'WG' in noise_type or 'Q' in noise_type:
                enhanced = signal.filtfilt(self.highpass_wgn[0],self.highpass_wgn[1],enhanced).astype('float64')
            elif 'B' in noise_type:
                enhanced = signal.filtfilt(self.highpass_bw[0],self.highpass_bw[1],enhanced).astype('float64')    
            
            self.out_folder = 'IIR'

        elif self.args.task=='FTS+IIR':
            enhanced = noisy

            if 'E' in noise_type and '+' not in noise_type:  
                enhanced, error = filtered_template_subtraction(enhanced,50)
                enhanced = signal.filtfilt(self.highpass_ecg[0],self.highpass_ecg[1],enhanced).astype('float64')    
            else:     
                if 'P' in noise_type:
                    enhanced = signal.filtfilt(self.notch[0],self.notch[1],enhanced).astype('float64')
                if 'm' in noise_type:
                    enhanced = signal.filtfilt(self.highpass_moa[0],self.highpass_moa[1],enhanced).astype('float64')
                elif 'WG' in noise_type or 'Q' in noise_type:
                    enhanced = signal.filtfilt(self.highpass_wgn[0],self.highpass_wgn[1],enhanced).astype('float64')
                elif 'B' in noise_type:
                    enhanced = signal.filtfilt(self.highpass_bw[0],self.highpass_bw[1],enhanced).astype('float64')

                if 'E' in noise_type and 'm' not in noise_type:
                    enhanced, error = filtered_template_subtraction(enhanced,50)
                    enhanced = signal.filtfilt(self.highpass_ecg[0],self.highpass_ecg[1],enhanced).astype('float64')   

            self.out_folder = 'FTS+IIR'

        elif self.args.task=='VMD':
            enhanced = noisy

            if 'E' in noise_type:
                enhanced = signal.filtfilt(self.highpass_ecg[0],self.highpass_ecg[1],enhanced).astype('float64')
                if 'P' in noise_type or 'WG' in noise_type:
                    enhanced = VMD_IIT_denoise(enhanced,'soft',noise_type).astype('float64')                        
            else:
                enhanced = VMD_IIT_denoise(enhanced,'soft',noise_type).astype('float64')
            
            self.out_folder = 'VMD'

        elif self.args.task=='EMD':
            enhanced = noisy
            enhanced = EMD_method(enhanced,stimulus,noise_type).astype('float64')
            self.out_folder = 'EMD'

        elif self.args.task=='EEMD':
            enhanced = noisy
            enhanced = EEMD_method(enhanced,stimulus,noise_type).astype('float64')
            self.out_folder = 'EEMD'
        
        elif self.args.task=='CEEMDAN':
            enhanced = noisy
            enhanced = CEEMDAN_method(enhanced,stimulus,noise_type).astype('float64')
            self.out_folder = 'CEEMDAN'

        self.used_time += time.time()-t_start

        # Evaluation metrics
        SNRin = cal_SNR(clean,noisy)
        SNRout = cal_SNR(clean,enhanced)
        SNRimp = SNRout-SNRin
        RMSE = cal_rmse(clean,enhanced)
        PRD = cal_prd(clean,enhanced)
        RMSE_ARV = cal_rmse(cal_ARV(clean),cal_ARV(enhanced))
        MF = cal_rmse(cal_MF(clean,stimulus),cal_MF(enhanced,stimulus))
        
        if self.args.task == 'denoise':
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{SNRin},{SNRimp},{SNRout},{loss},{RMSE},{PRD},{RMSE_ARV},{MF}\n')
        else:
            with open(self.score_path, 'a') as f1:
                f1.write(f'{outname},{SNRin},{SNRimp},{SNRout},{error},{RMSE},{PRD},{RMSE_ARV},{MF}\n')
        if self.output:
            emg_path = test_file.replace(f'{test_path}',f'./enhanced_data_for_comparison/{self.out_folder}/') 
            check_folder(emg_path)
            np.save(emg_path,enhanced)

            if self.args.output_latent:
                np.save(emg_path.replace('.npy','_map.npy'),feature_map)
                np.save(emg_path.replace('.npy','_transout.npy'),trans_out)
    
            
    def test(self):
        # load model
        #mkl.set_num_threads(1)
        
        print("best loss:",self.best_loss)
        if self.args.task == 'denoise':
            self.model.eval()
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model'])
            test_path = self.args.test_noisy if self.args.task=='denoise' else self.args.test_clean
        else:
            test_path = self.args.test_noisy
        
        test_folders = get_filepaths(test_path) if self.noise_type == 'all' else get_specific_filepaths(test_path,self.noise_type)
        
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f1:
            f1.write('Filename,SNRin,SNRimp,SNRout,Loss,RMSE,PRD,RMSE_ARV,MF\n')
        
        for test_file in tqdm(test_folders):
            self.write_score(test_file,test_path)
        
        print("Total time:", self.used_time)

        data = pd.read_csv(self.score_path)
        snrin_mean = data['SNRin'].to_numpy().astype('float').mean()
        snr_mean = data['SNRimp'].to_numpy().astype('float').mean()
        snrout_mean = data['SNRout'].to_numpy().astype('float').mean()
        loss_mean = data['Loss'].to_numpy().astype('float').mean()
        rmse_mean = data['RMSE'].to_numpy().astype('float').mean()
        prd_mean = data['PRD'].to_numpy().astype('float').mean()
        arv_mean = data['RMSE_ARV'].to_numpy().astype('float').mean()
        mf_mean = data['MF'].to_numpy().astype('float').mean()
        
        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(snrin_mean),str(snr_mean),str(snrout_mean),str(loss_mean),str(rmse_mean),str(prd_mean),str(arv_mean),str(mf_mean)))+'\n')
        
        print("Test SNRimp:",snr_mean,"Test RMSE",rmse_mean)

    def _train_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        pred = self.model(noisy)
        
        loss = self.criterion(clean, pred)
        snr_in = cal_SNR(clean, noisy, torch)
        snr_pred = cal_SNR(clean, pred, torch)

        self.train_loss += loss.item()
        self.train_snr += snr_pred-snr_in
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    
        
    def _val_step_mode_denoise(self, data):
        device = self.device
        noisy, clean = data
        noisy, clean = noisy.to(device).type(torch.float32), clean.to(device).type(torch.float32)
        pred = self.model(noisy)

        loss = self.criterion(pred, clean)
        snr_in = cal_SNR(clean, noisy, torch)
        snr_pred = cal_SNR(clean, pred, torch)

        self.val_snr += snr_pred-snr_in
        self.val_loss += loss.item()

    


    
