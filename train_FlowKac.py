
import os


import torch
from matplotlib import pyplot as plt
import numpy as np
import math
from lib.KR_model_GPU import flow_mapping, MLP_mapping
from lib.feynman_kac_tools import feynman_kac, SdeSampler
from lib.sde_init import init_sde
from tqdm import tqdm
import lib.utils as utils
import random


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    args = utils.get_arguments()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    utils.check_folder(args.save_dir)   
    
    logger = utils.get_logger(
        logpath=os.path.join(args.save_dir, "logs"), filepath=os.path.abspath(__file__), displaying = True
    )
    
    logger.info(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.NN_type == "NF":
        net_tnf = flow_mapping(input_dim = args.Dx, n_depth = args.n_depth, n_width = args.n_width, cond_dim=1, n_bins = args.n_bins ,rotation=False).to(device)
    elif args.NN_type == "MLP":
        net_tnf = MLP_mapping(input_dim = args.Dx, n_depth = args.n_depth, n_width = args.n_width).to(device)
    logger.info(net_tnf)

    logger.info(
        "Number of trainable parameters: {}".format(count_parameters(net_tnf))
    )
     
    t_input = torch.linspace(0, args.T_max, args.T_num).to(device)
    t_plot = torch.linspace(0, args.T_max, args.T_num).to(device)
    
    train_X = torch.zeros(args.X_num_train, args.T_num, args.Dx, device=device)
    train_T = t_input.view(1, args.T_num).expand(args.batch_size, args.T_num).to(device)
    
    
    ######################################
    #
    # SDE, sampler and optimize parameters
    #
    ######################################
    sde_FC = init_sde(args).to(device)
    
    # initialise the SDE sampler : exact, first_order or second_order
    sdeSampler = SdeSampler(sde = sde_FC, t_vec = t_input, args = args)
    
    optimizer = torch.optim.Adam((net_tnf.parameters()), lr=args.lr)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    
    fc_input = torch.zeros(args.X_num_train, device = device)
    
    x_plot  = np.linspace(args.x_0, args.x_1, args.N1)
    y_plot = np.linspace(args.x_0, args.x_1, args.N2)
    
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
    if args.resume is not None:
        checkpt = torch.load(os.path.join(args.save_dir,args.resume + ".pth"), weights_only = False)
        net_tnf.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
        if "last_Nadp" in checkpt.keys():
            args.begin_Nadp = checkpt["last_Nadp"]
            
    Tr_loss = utils.load_loss(args.save_dir) if args.resume is not None else []
    Te_loss = utils.load_test_loss(args.save_dir) if args.resume is not None else []
    best_loss = float("inf")
        
    for k in tqdm(range(args.begin_Nadp, args.N_adp+1)):
        
        fc_input = utils.make_fc_input(args, sde_FC, net_tnf, t_input, k)

        train_loader = utils.batch_loader(args, fc_input)
        epoch = args.epochs[k]
                
        for i in range(epoch):
            tr_loss = 0
            sampling_in_loop = False
            centroid = None

            if args.sto_taylor_order == "inf":
                sampling_in_loop = True
            elif args.adapt_x0 or args.augment_x0:
                sampling_in_loop = False
            else:
                sde_samples, sde_jacobian, sde_hessian = sdeSampler.sample()            
            
            net_tnf.train()

            for temp_i, input_data in enumerate(train_loader):
                
                if sampling_in_loop:
                    sde_samples, sde_jacobian, sde_hessian = sdeSampler.sample(input_data)
                elif args.adapt_x0:
                    centroid = input_data.mean(dim = 0, keepdim = True)
                    sde_samples, sde_jacobian, sde_hessian = sdeSampler.sample(centroid)
                

                input_data.requires_grad_()

                XX = input_data.view(input_data.shape[0], 1, args.Dx).expand(input_data.shape[0], args.T_num, args.Dx)
                TT = t_input.view(1, args.T_num).expand(input_data.shape[0], args.T_num)
                
                input_nf = torch.cat((XX.reshape(-1,args.Dx), TT.reshape(-1,1)), -1)

                if args.NN_type=="NF":
                    output, logdet = net_tnf(input_nf[:, :-1], cond_t=input_nf[:, -1:], logdet=torch.zeros_like(input_nf[..., 0:1]))
                    logdet = logdet + (-0.5 * output.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                    p = torch.exp(logdet)
                elif args.NN_type=="MLP":
                    p = net_tnf(input_nf)
                
                fc_solution = feynman_kac(sde_FC, sde_samples, input_data, t_input, args, sde_jacobian, sde_hessian, centroid)
                                
                loss = utils.flowkac_loss(args, k, p, fc_solution)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(net_tnf.parameters(), max_norm=args.max_grad_norm)

                optimizer.step()
                
                tr_loss = tr_loss + float(loss) * input_data.shape[0]
            
            tr_loss = tr_loss / len(train_loader.dataset)
            Tr_loss.append(tr_loss)
            #StepLR.step()
                        
            if tr_loss < best_loss:
                best_loss = tr_loss
                utils.save_best_model(net_tnf, os.path.join(args.save_dir), args)
            
            log_message = 'epoch:{} ===> train loss:{:.4e}, lr:{:.2e}'.format(i, tr_loss, optimizer.state_dict()['param_groups'][0]['lr'])
            logger.info(log_message)
            
            if (i+1) % args.test_freq == 0:
                
                net_tnf.eval()
                with torch.no_grad():
                    
                    KL_div_vec = []
                    relative_KL_div_vec = []
                    L2_err_vec = []
                    Test_loss = 0

                    for _,j in enumerate(args.time_index_test):
                        
                        test_data = utils.make_test_data(args, sde_FC, t_input[j])
                        
                        cond_t = t_input[j]*torch.ones_like(test_data[:,0:1])
                        
                        if args.NN_type=="NF":
                            output, logdet = net_tnf(test_data, cond_t = cond_t, logdet = torch.zeros_like(test_data[:,0:1]))
                            logdet = logdet + (-0.5 * output.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
                            p = torch.exp(logdet).detach().cpu().numpy()
                        elif args.NN_type == "MLP":
                            p = net_tnf(torch.cat((test_data, cond_t), dim = 1)).detach().cpu().numpy()
                            
                        p_true = sde_FC.density(t_input[j], test_data, args).detach().cpu().numpy()
                        
                        Test_loss = Test_loss + args.loss_scale*np.mean((p_true - p)**2)

                        temp_kl, temp_relative_kl = utils.compute_kl_div(p_true, p)
                        temp_l2_err = np.linalg.norm(p - p_true) / np.linalg.norm(p_true)
                        
                        KL_div_vec.append(temp_kl.item())
                        relative_KL_div_vec.append(temp_relative_kl)
                        L2_err_vec.append(float(temp_l2_err))
                    
                    Te_loss.append(Test_loss)
                    formatted_KL = [round(loss, 4) for loss in KL_div_vec]
                    formatted_L2 = [round(loss, 4) for loss in L2_err_vec]
                    logger.info("KL div : %s",formatted_KL)
                    logger.info("L2 errors : %s",formatted_L2)
                    
        utils.save_model(net_tnf, os.path.join(args.save_dir), args, optimizer, k)
        utils.save_model(net_tnf, os.path.join(args.save_dir), args, optimizer, k, "last")
        utils.save_loss(Tr_loss, args.save_dir)
        utils.save_test_loss(Te_loss, args.save_dir)
        
    plt.figure()
    plt.plot(Tr_loss)
    plt.semilogy()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("training loss")
    plt.savefig(os.path.join(args.save_dir,'training_loss.pdf'))

    plt.figure()
    plt.plot(Te_loss)
    plt.semilogy()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("test loss")
    plt.savefig(os.path.join(args.save_dir,'test_loss.pdf'))    

    logger.info("Training completed successfully")