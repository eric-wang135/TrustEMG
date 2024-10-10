import os, argparse, torch, random
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder, get_fold_num
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import sys

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True
#cudnn.enable =False

# assign gpu
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--task', type=str, default='denoise')  # evaluate/ denoise / IIR / FTS+IIR/ EMD/ VMD/ CEEMDAN

    parser.add_argument('--train_path', type=str, default='trainpt_E1_S40_Ch2_withSTI_seg2s_F1/')
    parser.add_argument('--train_noisy', type=str, default='noisy')
    parser.add_argument('--train_clean', type=str, default='clean')
    parser.add_argument('--test_noisy', type=str, default='./data_E2_S40_Ch11_withSTI_seg2s_F1/test/noisy/')
    parser.add_argument('--test_clean', type=str, default='./data_E2_S40_Ch11_withSTI_seg2s_F1/test/clean/')
    parser.add_argument('--writer', type=str, default='./train_log')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)  
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default=None)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='TrustEMGNet_RM') 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--worker_num', type=int, default=4)

    parser.add_argument('--resume' , action='store_true', default=False)
    parser.add_argument('--pretrained' , action='store_true', default=False)
    parser.add_argument('--pretrained_path', type=str, default='./save_model/denoise_classifier_02_epochs321_adam_bce_batch128_lr0.0001.pth.tar')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--val', action='store_true', default=True)
    parser.add_argument('--feature', action='store_true', default=False)
    parser.add_argument('--output', action='store_true', default=False)
    parser.add_argument('--data_aug', action='store_true', default=False)

    parser.add_argument('--noise_type', type=str, default='all') #options: E,P,B,P+B+E
    #parser.add_argument('--inputdim_fix', type=int, default=0)
    parser.add_argument('--output_latent', action='store_true', default=False)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
    print("Conda available:",torch.cuda.is_available())
    # get parameter
    args = get_args()

    fold = get_fold_num(args.test_noisy)
    
    if args.task == 'denoise': 
        # declair path
   
        checkpoint_path = f'./checkpoint/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}_{fold}.pth.tar'
        model_path = f'./save_model/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}_{fold}.pth.tar'
        
        score_path = f'./Result/{args.task}_{args.model}_epochs{args.epochs}' \
                        f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                        f'lr{args.lr}_{fold}.csv'
        # tensorboard
        writer = SummaryWriter(args.writer)

        # import model from its directory and create a model
        exec (f"from model.{args.model.split('_')[0]} import {args.model} as model")
    
        if args.pretrained:
            pretrained_state_dict = torch.load(args.pretrained_path)
            model = model(pretrained_state_dict['model'])
        else:
            model = model()
        
        model, epoch, best_loss, optimizer, criterion, device, lr_scheduler = Load_model(args,model,checkpoint_path,model)
        loader = Load_data(args) if args.mode == 'train' else 0
        print("Establish trainer")
        Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, lr_scheduler,
                        criterion, device, loader, writer, model_path, score_path,args)
    else:
        score_path = f'./Result/{args.task}_{fold}.csv'
        
        Trainer = Trainer(score_path=score_path, args=args)

    try:
        if args.mode == 'train':
            print("Training start")
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }

        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('epoch:',epoch)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
