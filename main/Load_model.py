import torch, os, random
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from util import get_filepaths
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

def count_parameters(model):
    # count parameters unfixed
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def weights_init(m):
    #initialize m's weight and bias
    if isinstance(m, nn.Conv1d): #or isinstance(m, nn.Linear):
        print(m)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        
def load_checkoutpoint(model,optimizer,lr_scheduler,checkpoint_path):

    if os.path.isfile(checkpoint_path):
        model.eval()
        print("=> loading checkpoint '{}'".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
        
        return model,epoch,best_loss,optimizer,lr_scheduler
    else:
        raise NameError(f"=> no checkpoint found at '{checkpoint_path}'")

def Load_model(args,model,checkpoint_path,param):
    
    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    
    # loss function type
    criterion = {
        'mse'     : nn.MSELoss(),
        'l1'      : nn.L1Loss(),
        'l1smooth': nn.SmoothL1Loss(),
        }

    criterion = criterion[args.loss_fn].to(device)

    optimizers = {
        'adam'    : Adam(param.parameters(),lr=args.lr, weight_decay=0),
        'SGD'     : SGD(param.parameters(), lr=args.lr, momentum=0.9)}
        
    optimizer = optimizers[args.optim]

    if args.lr_scheduler != '':
        lr_scheduler = MultiStepLR(optimizer,[3, 30, 40], gamma=0.1)
    else:
        lr_scheduler = None

    if args.resume:
        model,epoch,best_loss,optimizer,lr_scheduler = load_checkoutpoint(model,optimizer,lr_scheduler,checkpoint_path)
        
    else:
        epoch = 0
        best_loss = 10
        model.apply(weights_init)
    
    para = count_parameters(model)
    print(f'Num of model parameter : {para}')
    
    return model,epoch,best_loss,optimizer,criterion,device,lr_scheduler

def Load_data(args):
    # Seperate dataset into training and validation dataset 
    if args.task=='denoise':
        if args.val:
                train_paths = get_filepaths(f'{args.train_path}/noisy','.pt')
                val_paths = get_filepaths(f'{args.train_path}/val','.pt')  
        else:
            filepaths = get_filepaths(f'{args.train_path}/noisy','.pt') 
            train_paths,val_paths = train_test_split(filepaths,test_size=0.1,random_state=999)
        
        if args.data_aug:
            train_dataset, val_dataset = CustomDataset(train_paths,args.train_path+args.train_clean,aug=True), CustomDataset(val_paths,args.train_path+args.train_clean)
        else:
            train_dataset, val_dataset = CustomDataset(train_paths,args.train_path+args.train_clean), CustomDataset(val_paths,args.train_path+args.train_clean)

    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.worker_num,shuffle=True, pin_memory=True),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.worker_num,pin_memory=True)
    }
    
    return loader

class CustomDataset(Dataset):
    def __init__(self, paths,clean_path=None,aug=False):   # initial logic happens like transform
        self.n_paths = paths
        self.clean_path = clean_path
        self.aug = aug
        if clean_path:
            self.c_paths = [os.path.join(clean_path,noisy_path.split('/')[-1]) for noisy_path in paths]
        
    def __getitem__(self, index):
        noisy = torch.load(self.n_paths[index])
        if self.aug and random.random()>0.5:
            noisy = self.random_zero(noisy)
        if self.clean_path:
            clean = torch.load(self.c_paths[index])
            return (noisy,clean)
        return noisy

    def __len__(self):  # return count of sample we have        
        return len(self.n_paths)

    def random_zero(_, noisy):
        data_len = noisy.shape[0]
        start = random.randint(0,data_len-data_len//10-1)
        end = start + data_len//10 + 1
        noisy[start:end] = 0
        return noisy

        
        
    
