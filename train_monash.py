
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import random


import os
import argparse

from itertools import product

from models.ls4 import VAE
from tqdm.auto import tqdm

from pathlib import Path
from omegaconf import OmegaConf
from datasets import parse_datasets
import wandb
import sklearn

import matplotlib.pyplot as plt
from metrics import compute_all_metrics

def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training', add_help=False)
    # Optimizer
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    # Dataloader
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers to use for dataloader')
    # Model
    
    # parser.add_argument('--config', default='./configs/monash/vae_nn5daily.yaml', type=str, help='Number of layers')
    # parser.add_argument('--config', default='./configs/monash/vae_solarweekly.yaml', type=str, help='Number of layers')
    parser.add_argument('--config', default='./configs/monash/vae_temperature_rain.yaml', type=str, help='Number of layers')
    # parser.add_argument('--config', default='./configs/monash/vae_fred_md.yaml', type=str, help='Number of layers')
    
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test', '-t', action='store_true', help='Test')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--log_dir', default="./outputs", help='debug')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--wandb_entity', default='', help='Your Wandb account')

    return parser



###############################

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.
    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.
    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler



###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(model, ema_model, ema_start_step, device, 
          trainloader, optimizer, scheduler, step):
    model.train()
    train_loss = 0
    mse = 0
    kld = 0
    nll = 0
    ce = 0

    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (data, masks) in pbar:
        step += 1
        data = data.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        loss , log_info = model(data, None, masks,
                                plot=batch_idx==0, sum=False)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mse += log_info['mse_loss']
        kld += log_info['kld_loss']
        nll += log_info['nll_loss']
        ce += log_info.get('ce_loss', 0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | KLD: %.3f | NLL: %.3f | MSE: %.7f | CE: %.3f' %
            (batch_idx, len(trainloader), train_loss / (batch_idx + 1), kld / (batch_idx + 1),
             nll / (batch_idx + 1), mse / (batch_idx + 1), ce / (batch_idx + 1))
        )

        wandb.log({f'train_{k}':v / (batch_idx + 1) for k, v in dict(log_info).items()})
        
        if ema_model and step > ema_start_step:
            ema_model.update_parameters(model)
            

    scheduler.step(train_loss / (batch_idx + 1))
    return step
    
    
@torch.no_grad()
def eval(model, device, epoch, dataloader, decode_func, log_dir, savenpy=False):


    img = os.path.join(log_dir, 'images')
    epoch_dir = os.path.join(img, f'{epoch:03d}')

    if not os.path.isdir(img):
        os.mkdir(img)
        
    
    if not os.path.isdir(epoch_dir):
        os.mkdir(epoch_dir)

    model.eval()
    mse_loss = 0
    mse_noscale_loss = 0
    nll_loss = 0
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        model.module.setup_rnn()
    else:
        model.setup_rnn()

    pbar = tqdm(enumerate(dataloader))
    for batch_idx, (data, masks)  in pbar:
        data = data.to(device)
        masks = masks.to(device)
        if isinstance(model, torch.nn.DataParallel)  or isinstance(model, torch.optim.swa_utils.AveragedModel):
            recon  = model.module.reconstruct(data, None, masks=masks,
                                                       get_full_nll=False)
        else:
            recon = model.reconstruct(data, None,  masks=masks,
                                                get_full_nll=False)
        
        
        recon_scaled = decode_func(recon)
        data_scaled = decode_func(data)
        
        mask_sum = masks.sum(1)
        mask_sum[mask_sum==0] = 1
        mse = (
            ((recon_scaled - data_scaled).pow(2) * masks).sum(1) / mask_sum
        ).mean()
        
        mse_noscale = (
            ((recon - data).pow(2) * masks).sum(1) / mask_sum
        ).mean()

        ## likelihood
        
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
            full_nll = model.module.full_nll(data, None, torch.ones_like(data),)
        else:
            full_nll = model.full_nll(data, None, torch.ones_like(data),)

        mse_loss += mse.detach().cpu().item()
        nll_loss += full_nll.detach().cpu().item()
        mse_noscale_loss+= mse_noscale.detach().cpu().item()


    #gen
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        gen = model.module.generate(16, data.shape[1], device=data.device)
    else:
        gen =  model.generate(16, data.shape[1], device=data.device)

    gen = (gen).cpu().numpy()
    for k in range(min(gen.shape[0], 8)): 
        filename = f'{epoch_dir}/{k:02d}_gen.png'
        plt.plot(gen[k], c='black', alpha=0.6)
        plt.savefig(filename)
        plt.close()
        
    if savenpy:
        np.save(f'{epoch_dir}/gen.npy', gen)
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.optim.swa_utils.AveragedModel):
        recon  = model.module.reconstruct(data, None, masks=masks,
                                                    get_full_nll=False)
    else:
        recon = model.reconstruct(data, None,  masks=masks,
                                            get_full_nll=False)
    
    recon = (recon).cpu().numpy()
    data = (data).cpu().numpy()
    for k in range(min(recon.shape[0], 8)): 
        plt.plot(recon[k], c='black', alpha=0.6)
        plt.savefig(f'{epoch_dir}/{k:02d}_recon.png')
        plt.close()
        plt.plot(data[k], c='black', alpha=0.6)
        plt.savefig(f'{epoch_dir}/{k:02d}_x.png')
        plt.close()
    
    if savenpy:
        np.save(f'{epoch_dir}/data.npy', data)
    p = 'TEST MSE: %.8f TEST MSE No Scale: %.8f  FULL NLL: %.8f' % \
        (mse_loss / (batch_idx + 1), mse_noscale_loss/(batch_idx + 1), nll_loss / (batch_idx + 1))
    print(p)
    wandb.log({'eval_mse': mse_loss / (batch_idx + 1),
               'eval_mse_noscale':  mse_noscale_loss/(batch_idx + 1),
               'eval_full_nll': nll_loss / (batch_idx + 1)})

    return mse_loss / (batch_idx + 1)




def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = OmegaConf.load(args.config)
    args.dataset = config.data.dataset
    args.batch_size = config.optim.batch_size
    args.doc = config.doc
    
    name = f'{args.doc}'
    if args.wandb_entity == '':
        print('No logging to wandb')
    wandb.init(project="latent_s4", entity=args.wandb_entity,
               name=name, config={**vars(args), **pd.json_normalize(config, sep='_')}, tags=[args.dataset],
            mode='disabled' if args.debug or args.wandb_entity == '' else 'online')
    wandb.save(args.config)
    
    # Data
    print(f'==> Preparing {args.dataset} data..')
    data_objs = parse_datasets(config.data, batch_size=args.batch_size, device=torch.device("cpu"))
    # Dataloaders
    trainloader = data_objs['train_dataloader']
    valloader =  data_objs['test_dataloader']
    decode_func = data_objs['decode_func']
    n_labels = data_objs.get('n_labels', 1)
    config.model.n_labels = n_labels
    
    # Model
    print('==> Building model..')
    model = VAE(config.model)

    model = model.to(device)
    
    use_ema = config.optim.get('use_ema', False)
    if use_ema:
        ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            config.optim.lamb * averaged_model_parameter + (1 - config.optim.lamb) * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
        ema_start_step =  config.optim.start_step
    else:
        ema_model = None
        ema_start_step = 0

    optimizer, scheduler = setup_optimizer(
        model, lr=config.optim.lr, weight_decay=config.optim.weight_decay, epochs=config.optim.epochs
    )


    best_marginal = float('inf')  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    step = 0
    metrics_log = {
        'mse': [],
        'clf_score': [],
        'marginal_score': [],
        'predictive_score': []
    }

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    log_dir =f'{args.log_dir}/{args.dataset}/{args.doc}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if args.resume:
        # Load checkpoint.
        ckpt = os.path.join(log_dir, 'checkpoints')
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(ckpt), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(ckpt, 'ckpt.pth'))
        model.load_state_dict(checkpoint['model'])
        metrics_log = checkpoint['stats']
        start_epoch = checkpoint['epoch']
        step = checkpoint['step']
        if use_ema and step > ema_start_step:
            ema_model.load_state_dict(checkpoint['ema_model'])


    if not args.test:
        pbar = tqdm(range(start_epoch, start_epoch + config.optim.epochs), disable=not args.debug)
        for epoch in pbar:
            if epoch == 0 or args.resume:
                p = 'Epoch: %d' % (epoch)
                if not args.debug:
                    print(p)
                else:
                    pbar.set_description(p)

            step = train(model, ema_model, ema_start_step, device, trainloader, optimizer, scheduler, step)
            
            if step > ema_start_step:
                eval_model = ema_model or model
            else:
                eval_model = model
                
            if epoch % config.optim.eval_iter == 0 and epoch > 0:
                val_mse = eval(eval_model , device, epoch, valloader, decode_func,log_dir=log_dir)
                metrics_log['mse'].append(val_mse)
                if len(metrics_log['mse']) > 5:
                    metrics_log['mse'].pop(0)
                if not metrics_log['clf_score']:
                    p = 'Epoch: %d | Val mse: %1.3f' % (epoch, val_mse,)
                else:
                    p = 'Epoch: %d | Val mse: %1.3f' % (epoch, val_mse,)
                    for v in metrics_log:
                        p += ' | %s: (%1.3f, %1.3f +- %1.3f)' % (v, metrics_log[v][-1],np.mean(metrics_log[v]), np.std(metrics_log[v]))
    
                if not args.debug:
                    print(p)
                else:
                    pbar.set_description(p)
                
            if epoch % config.optim.metric_iter == 0 and epoch > 0:
                scores = compute_all_metrics(eval_model, valloader, setup_optimizer, device,
                                             nn.Sigmoid() if args.dataset == 'temperature_rain' else nn.Identity())
                for v in scores:
                    metrics_log[v].append(scores[v])
                    if len(metrics_log[v]) > 5:
                        metrics_log[v].pop(0)
                p = ''
                for v in metrics_log:
                    p += ' | %s: (%1.3f, %1.3f +- %1.3f)' % (v, metrics_log[v][-1], np.mean(metrics_log[v]), np.std(metrics_log[v]))
    
                if not args.debug:
                    print(p)
                else:
                    pbar.set_description(p)
                    
                
                wandb.log(scores)
                # Save checkpoint.
                marginal = scores['marginal_score']
                if marginal < best_marginal:
                    
                    ckpt = os.path.join(log_dir, 'checkpoints')
                    if not os.path.isdir(ckpt):
                        os.mkdir(ckpt)
                    state = {
                        'model': model.state_dict(),
                        'stats': metrics_log,
                        'epoch': epoch,
                        'step': step
                    }
                    if ema_model and step > ema_start_step:
                        state['ema_model'] = ema_model.state_dict()
                    if not os.path.isdir(ckpt):
                        os.mkdir(ckpt)
                    torch.save(state, os.path.join(ckpt, 'ckpt.pth'))
                    best_marginal = marginal
    if step > ema_start_step:
        eval_model = ema_model or model
    else:
        eval_model = model
    eval(eval_model , device, -1, valloader, decode_func, log_dir=log_dir, savenpy=True)
    scores = compute_all_metrics(eval_model, valloader, setup_optimizer, device,
                                 nn.Sigmoid() if args.dataset == 'temperature_rain' else nn.Identity())
    p = 'Final'
    for v in scores:
        p += ' | %s: %1.3f' % (v, scores[v])
    print(p) 

if __name__ == '__main__':
    args = get_parse_args()
    args = args.parse_args()
    if args.seed:
        torch.random.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    main(args)
