# -*- coding: utf-8 -*-

import logging
import pickle
import torch
import os
import numpy as np
from scipy.stats import entropy
import argparse
from sklearn.cluster import KMeans


def get_arguments():
    parser = argparse.ArgumentParser(description="SDE parameters")

    parser.add_argument("--Dx", type=int, default=1, help="Dimension of the state variable (default: 1)")
    parser.add_argument("--W_Dx", type=int, default=1, help="Dimension of the Brownian motion (default: 1)")
    parser.add_argument("--epoch", type=int, default=20, help="Total number of epochs (default: 20)")
    parser.add_argument("--epochs", type=str, default= "20, 40, 60, 80", help="Epochs for each stage (default: '20, 40, 60, 80')")
    parser.add_argument("--N_adp", type=int, default=5, help="Number of adaptations (default: 5)")
    parser.add_argument("--n_depth", type=int, default=4, help="Depth of neural networks (default: 4)")
    parser.add_argument("--X_num_train", type=int, default=10000, help="Number of space training points (default: 10000)")
    parser.add_argument("--T_num", type=int, default=21, help="Number of time training points (default: 21)")
    parser.add_argument("--multiplier_T_num", type=int, default=5, help="Multiplier of time vec length to go from T_num to T_num x Multiplier. Increase sampling accuracy")
    parser.add_argument("--n_width", type=int, default=32, help="Width of neural networks (default: 32)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default: X_num_train/10)")
    parser.add_argument("--N_sample", type=int, default=300, help="Number of samples (default: 300)")
    parser.add_argument("--save_dir", type=str, default="experiments/sde", help="Directory to save results (default: experiments/sde)")
    parser.add_argument("--sde_name", type=str, default="gbm_1d", help="Name of the SDE model (default: gbm_1d)")
    parser.add_argument("--sto_taylor_order", type=str, choices=["first_order","first_order_exact", "second_order", "second_order_exact","inf"], help="order of stochastic flow approximation through Taylor expansion (default: first)")
    parser.add_argument("--x_0", type=float, default=0.1, help="Initial value of x (default: 0.1)")
    parser.add_argument("--x_1", type=float, default=2, help="Final value of x (default: 2)")
    parser.add_argument("--x_num_test", type=int, default=100, help="Number of test points per dimension")
    parser.add_argument("--T_max", type=float, default=1, help="final value of time vector (default: 1)")
    parser.add_argument("--time_index_test", type=str, default="0, 5, 10, 15", help="index of time values to test")
    
    parser.add_argument("--N1", type=int, default=100, help="x-dim for plot")
    parser.add_argument("--N2", type=int, default=100, help="y-dim for plot")
    parser.add_argument("--N3", type=int, default=50, help="x3-dim for plot")
    parser.add_argument("--N4", type=int, default=50, help="x4-dim for plot")
    parser.add_argument("--n_bins", type=int, default=0, help="non linear layer bins")
    
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--begin_Nadp", type=int, default=0)
    
    parser.add_argument("--torchsde_method", type=str, choices=["euler","milstein"], help="numerical integration method")
    parser.add_argument("--test_freq", type=int, default=20, help="frequence to compute test metrics and plot")
    parser.add_argument("--N_sample_test", type=int, default=20000, help="number of test points to sample for SDEs with dim higher than 4")
    parser.add_argument("--train_data_type", type=str, choices=["uniform","init"], default = "uniform", help="train set sampling method")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_scale", type=float, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1e10, help="Max norm of gradients (default is extremely high to avoid any clipping)")

    parser.add_argument("--reverse_nf_support", type=eval,default = False,  help="reverse NF for dynamic density support")
    parser.add_argument("--min_density_support", type=eval,default = False, help="force min value for dynamic density support (avoid negative values for GBM)")
    parser.add_argument("--index_dyn_support", type=str, default="0,0,0,0")

    parser.add_argument("--adapt_x0", type=eval, default = False, help = "adapt x0 to each batch")
    parser.add_argument("--augment_x0", type=eval, default = False, help = "augment x0 with stochastic sampling trick")
    parser.add_argument("--aug_size", type=int, default = 1, help = "size of data augmentation")
    parser.add_argument("--aug_range", type=float, default = 1, help = "range of data augmentation around the original data point")
    parser.add_argument("--importance_sampling", type=eval, default=False, choices=[True, False])

    parser.add_argument("--seed", type=int, default=None, help="seed to ensure deterministic behavior")
    parser.add_argument("--NN_type", type=str, choices=["NF","MLP"])

    args, unknown = parser.parse_known_args()
    
    # Convert the string to a list of integers
    if args.time_index_test:
        args.time_index_test = [int(i_plot) for i_plot in args.time_index_test.split(',')]
    else:
        args.time_index_test = []
        
    if args.epochs:
        args.epochs = [int(epoch) for epoch in args.epochs.split(',')]
    else:
        args.epochs = []

    if args.index_dyn_support:
        args.index_dyn_support = [int(index_i) for index_i in args.index_dyn_support.split(',')]
    else:
        args.index_dyn_support = []
        
    return args

def get_logger(
        logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def compute_kl_div(P, Q):
    
    # Avoid division by zero and log of zero by adding a small value (epsilon)
    epsilon = 1e-10
    P = np.clip(P, epsilon, 1)
    Q = np.clip(Q, epsilon, 1)

    # Compute the KL divergence
    #kl_divergence = np.sum(P * np.log(P / Q))
    kl_divergence = entropy(P, Q)
    
    #relative_kl_divergence = kl_divergence/(-np.sum( P * np.log(P) ) )
    relative_kl_divergence = kl_divergence/entropy(P)
    

    return kl_divergence, relative_kl_divergence


def save_object(obj, path):
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
        

def save_model(model, save_path, args, optimizer, Nadp, step=None):
    
    if step == "last":
        total_save_path = os.path.join(save_path,'last.pth')
    else:
        total_save_path = os.path.join(save_path,'full_k'+ str(Nadp) + '.pth')
        
    torch.save(
        {
            "args": args,
            "state_dict": model.state_dict()
            if torch.cuda.is_available()
            else model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "last_Nadp": Nadp + 1,
        },
        total_save_path,
    )
    
def check_folder(save_path):
    folder_path = os.path.dirname(save_path + "/")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_loss(loss_list, save_path):
    np.save(os.path.join(save_path, 'loss_history.npy'), np.array(loss_list))

def save_test_loss(loss_list, save_path):
    np.save(os.path.join(save_path, 'test_loss_history.npy'), np.array(loss_list))

def load_loss(save_path):
    loss_file = os.path.join(save_path, 'loss_history.npy')
    if os.path.exists(loss_file):
        return list(np.load(loss_file))
    return []

def load_test_loss(save_path):
    loss_file = os.path.join(save_path, 'test_loss_history.npy')
    if os.path.exists(loss_file):
        return list(np.load(loss_file))
    return []

def make_fc_input(args, sde_FC, net_tnf, t_input, k):
    """Generates training data or reverse NF to dynamically obtain the converging support."""
    if k == 0:
        return make_training_data(args, sde_FC)
    
    if args.reverse_nf_support:
        with torch.no_grad():
            XX_ref = torch.randn(args.X_num_train, 1, args.Dx, device=args.device).expand(args.X_num_train, args.T_num, args.Dx)
            TT_ref = t_input.view(1, args.T_num).expand(args.X_num_train, args.T_num)

            tnf_input_ref = torch.cat((XX_ref.reshape(-1, args.Dx), TT_ref.reshape(-1, 1)), dim=-1)
            # reverse temporal NF to get dynamically converging density support
            fc_input = net_tnf(tnf_input_ref[:, :-1], cond_t=tnf_input_ref[:, -1:], reverse=True).clone()
            fc_input = fc_input.view(args.X_num_train, args.T_num, args.Dx)[:, args.index_dyn_support[k], :]

            if args.min_density_support: # enforce a minimum value for densities with constraints (i.e. GBM)
                fc_input = torch.maximum(fc_input, torch.tensor(args.x_0, device=args.device))
    else:
        fc_input = make_training_data(args, sde_FC)

    return fc_input

def make_training_data(args, sde_FC):
    if args.train_data_type == "uniform":
        train_data = torch.rand(args.X_num_train, args.Dx,device=args.device) * (args.x_1 - args.x_0) + args.x_0
    elif args.train_data_type == "init":
        train_data = sde_FC.sample_test_data(torch.zeros(1, device = args.device), args.X_num_train).to(args.device)
    else:
        raise NotImplementedError
    return train_data

def make_test_data(args, sde_FC, t):
    
    if args.Dx > 4:
        return sde_FC.sample_test_data(t, args.N_sample_test).to(t)
    
    temp_data = torch.linspace(args.x_0, args.x_1, args.x_num_test)
    temp_data = torch.meshgrid(*(temp_data,) * args.Dx, indexing='ij')
    test_data = torch.stack(temp_data, dim=-1).reshape(-1, args.Dx)
        
    return test_data.to(t)


def save_best_model(net_tnf, save_path, args):
    
            
    dict_save_path = os.path.join(save_path, 'best_Nadap_' + str(args.N_adp) + '_Depth_' + str(args.n_depth) + '_Width_' + str(args.n_width) + '.pth')
    torch.save(net_tnf.state_dict(), dict_save_path)
    

# Function to save checkpoints
def save_checkpoint(model, optimizer, epoch, save_dir, k, prefix="checkpoint"):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'k': k
    }
    checkpoint_path = os.path.join(save_dir, f"{prefix}_k{k}_epoch{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    #print(f"Checkpoint saved at {checkpoint_path}")

# Function to load checkpoints
def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch'] # start at the next epoch
    k = checkpoint['k']
    return model, optimizer, epoch, k

# Check if there is a checkpoint to resume from
def resume_from_checkpoint(save_dir, k):
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(f"checkpoint_k{k}")]
    if checkpoint_files:
        latest_checkpoint = sorted(checkpoint_files)[-1]
        return os.path.join(save_dir, latest_checkpoint)
    return None


# Function to save the global loss history to a file
def save_loss_history(loss_history, filepath):
    np.save(filepath, loss_history)

# Function to load the global loss history from a file
def load_loss_history(filepath, args):
    total_epochs = sum(args.epochs[:(args.N_adp+1)])
    
    if os.path.exists(filepath):
        return np.load(filepath)
    return np.full(total_epochs, np.nan)


def flowkac_loss(args, k, p_NF, p_FK):
    

    square_error = torch.square(p_NF - p_FK)

    if k>=1 and args.reverse_nf_support and args.importance_sampling:
        eps = torch.tensor(1e-12, device = p_NF.device) # small constant to avoid dividing by zero
        return args.loss_scale*torch.mean( square_error/(p_NF + eps))
        
    return args.loss_scale*torch.mean(square_error)


def augment_data(args, data_input):
    N, d = data_input.shape
    
    # uniform samples in [-range_aug, range_aug]
    noise = torch.rand(N, args.aug_size, d,device=args.device) * (args.aug_range*2) - args.aug_range
    
    augmented_train_data = data_input.unsqueeze(1).expand(N, args.aug_size, d) + noise
    
    return augmented_train_data.reshape(-1,d)

def batch_loader(args, fc_input):
    if args.adapt_x0:
        num_of_batch = args.X_num_train // args.batch_size
        kmeans = KMeans(n_clusters=num_of_batch, n_init=10)
        labels = kmeans.fit_predict(fc_input.detach().cpu().numpy())
        
        batches = []
        for i in range(num_of_batch):
            batch = fc_input[torch.tensor(labels) == i]
            batches.append(batch)
            
        train_loader = torch.utils.data.DataLoader(torch.cat(batches, dim=0), batch_size=args.batch_size, shuffle = False)
        
    else:
        
        train_loader = torch.utils.data.DataLoader(fc_input, batch_size=args.batch_size, shuffle = True)
        
    
    return train_loader