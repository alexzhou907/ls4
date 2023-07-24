
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

from properscoring import crps_gaussian

def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training', add_help=False)
    # Optimizer
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay')
    # Dataloader
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers to use for dataloader')
    # Model
    # parser.add_argument('--config', default='./configs/interpolation/vae_climate_interp.yaml', type=str, help='Number of layers')
    # parser.add_argument('--config', default='./configs/interpolation/vae_physionet_interp.yaml', type=str, help='Number of layers')
    # parser.add_argument('--config', default='./configs/extrapolation/vae_climate_extrap.yaml', type=str, help='Number of layers')
    parser.add_argument('--config', default='./configs/extrapolation/vae_physionet_extrap.yaml', type=str, help='Number of layers')

    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--test', '-t', action='store_true', help='Test')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--log_dir', default="./outputs", help='debug')
    parser.add_argument('--seed', default=42, type=int)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

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
def train(model, device, trainloader, optimizer, extrap=False, step=0):
    model.train()
    train_loss = 0
    mse = 0
    kld = 0
    nll = 0
    ce = 0

    pbar = tqdm(enumerate(trainloader))
    for batch_idx, data in pbar:
        step += 1
        if extrap:
            data['observed_data'] = \
                torch.cat([data['observed_data'], data['data_to_predict']], dim=1)
            data['observed_tp'] = \
                torch.cat([data['observed_tp'], data['tp_to_predict']], dim=0)
            data['observed_mask'] = \
                torch.cat([torch.zeros_like(data['observed_mask']), data['mask_predicted_data']], dim=1)

        data['observed_data']=data['observed_data'].to(device)
        data['observed_tp']=data['observed_tp'].to(device)
        data['observed_mask']=data['observed_mask'].to(device)
        data['labels'] = data['labels'].to(device)

        optimizer.zero_grad()
        loss , log_info = model(data['observed_data'],
                                data['observed_tp'],
                                data['observed_mask'],
                                data['labels'],
                                plot=batch_idx==0)
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


    return step
@torch.no_grad()
def eval(model, device, epoch, dataloader, best_mse, log_dir, classify=None, extrap=False):


    img = os.path.join(log_dir, 'images')


    if not os.path.isdir(img):
        os.mkdir(img)

    model.eval()
    mse_loss = 0
    crps_loss = 0
    nll_loss = 0
    if isinstance(model, torch.nn.DataParallel):
        model.module.setup_rnn()
    else:
        model.setup_rnn()

    label_all, pred_all, mask_all =[], [], []

    pbar = tqdm(enumerate(dataloader))
    for batch_idx, data in pbar:
        data['observed_data'] = data['observed_data'].to(device)
        data['observed_tp'] = data['observed_tp'].to(device)
        data['observed_mask'] = data['observed_mask'].to(device)

        data['data_to_predict'] = data['data_to_predict'].to(device)
        data['tp_to_predict'] = data['tp_to_predict'].to(device) if extrap else None
        data['mask_predicted_data'] = data['mask_predicted_data'].to(device)
        data['labels'] = data['labels'].to(device)

        if isinstance(model, torch.nn.DataParallel):
            recon  = model.module.reconstruct(data['observed_data'],
                            data['observed_tp'], data['tp_to_predict'], masks=data['mask_predicted_data'],
                                                       get_full_nll=False)
            sigma = model.module.config.sigma
        else:
            recon = model.reconstruct(data['observed_data'],
                            data['observed_tp'], data['tp_to_predict'], masks=data['mask_predicted_data'],
                                                get_full_nll=False)
            sigma = model.config.sigma

        mask_sum = data['mask_predicted_data'].sum(1)
        mask_sum[mask_sum==0] = 1
        mse = (
            ((recon - data['data_to_predict']).pow(2)*data['mask_predicted_data']).sum(1) / mask_sum
        ).mean()
        
        crps = ((crps_gaussian(data['data_to_predict'].detach().cpu().numpy(), mu=recon.detach().cpu().numpy(), sig=sigma)*data['mask_predicted_data'].detach().cpu().numpy()).sum(1)/ mask_sum.detach().cpu().numpy()).mean()

        ## likelihood
        if extrap:
            x = \
                torch.cat([data['observed_data'], data['data_to_predict']], dim=1)
            tp = \
                torch.cat([data['observed_tp'], data['tp_to_predict']], dim=0)
            m = \
                torch.cat([torch.zeros_like(data['observed_mask']), data['mask_predicted_data']], dim=1)
        else:
            x=  data['observed_data']
            tp = data['observed_tp']
            m = data['observed_mask']
        if isinstance(model, torch.nn.DataParallel):
            full_nll = model.module.full_nll(x, tp, m)
        else:
            full_nll = model.full_nll(x, tp, m)

        mse_loss += mse.detach().cpu().item()
        nll_loss += full_nll.detach().cpu().item()
        crps_loss += crps
        if classify:

            if extrap:
                data['observed_data'] = \
                    torch.cat([data['observed_data'], data['data_to_predict']], dim=1)
                data['observed_tp'] = \
                    torch.cat([data['observed_tp'], data['tp_to_predict']], dim=0)
                data['observed_mask'] = \
                    torch.cat([data['observed_mask'], data['mask_predicted_data']], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                pred = model.module.predict(data['observed_data'],
                                            data['observed_tp'],
                                            data['observed_mask'])
            else:
                pred = model.predict(data['observed_data'],
                                     data['observed_tp'],
                                     data['observed_mask']
                                     )
            pred_all.append(pred.detach())
            label_all.append(data['labels'].squeeze(-1).detach())

            if classify == 'per_tp':
                mask_all.append((data['observed_mask'].sum(-1)>0).detach())


    p = 'TEST MSE: %.8f FULL NLL: %.8f CRPS: %.8f  ' % \
        (mse_loss / (batch_idx + 1), nll_loss / (batch_idx + 1),  crps_loss / (batch_idx + 1))
    print(p)
    wandb.log({'eval_mse': mse_loss / (batch_idx + 1),
               'eval_full_nll': nll_loss / (batch_idx + 1),
               'crps':  crps_loss / (batch_idx + 1)})

    if classify:
        label_all = torch.cat(label_all, dim=0)
        pred_all = torch.cat(pred_all, dim=0)
        if classify == 'per_seq':
            idx_not_nan = ~torch.isnan(label_all)

            acc = sklearn.metrics.accuracy_score(
                label_all[idx_not_nan].cpu().numpy(),
                pred_all[idx_not_nan].cpu().numpy())

            p = f'Accuracy: {acc:>.8f}'
            print(p)
            wandb.log({'test_accuracy': acc})
            auc = sklearn.metrics.roc_auc_score(label_all[idx_not_nan].cpu().numpy().reshape(-1),
                                                pred_all[idx_not_nan].cpu().numpy().reshape(-1))

            p = f'AUC: {auc:>.8f}'
            print(p)

            wandb.log({'test_AUC': auc})
        elif classify == 'per_tp':

            mask_all = torch.cat(mask_all, dim=0)
            _, label_all = torch.max(label_all, -1)
            acc = sklearn.metrics.accuracy_score(
                label_all[mask_all].cpu().numpy(),
                pred_all[mask_all].cpu().numpy())

            p = f'Accuracy: {acc:>.8f}'
            print(p)
            wandb.log({'test_accuracy': acc})

    # Save checkpoint.
    mse = mse_loss / (batch_idx + 1)

    return mse, best_mse




def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = OmegaConf.load(args.config)
    args.dataset = config.data.dataset
    args.batch_size = config.optim.batch_size
    args.doc = config.doc

    # Data
    print(f'==> Preparing {args.dataset} data..')
    data_objs = parse_datasets(config.data, batch_size=args.batch_size, device=torch.device("cpu"))
    # Dataloaders
    trainloader = data_objs['train_dataloader']
    valloader =  data_objs['test_dataloader']
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

    name = f'{args.doc}'
    if args.wandb_entity == '':
        print('No logging to wandb')
    wandb.init(project="latent_s4", entity=args.wandb_entity,
               name=name, config={**vars(args), **pd.json_normalize(config, sep='_')}, tags=[args.dataset],
            mode='disabled' if args.debug or args.wandb_entity == '' else 'online')
    wandb.save(args.config)
    best_mse = float('inf')  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    step = 0
    log_dir =f'{args.log_dir}/{args.dataset}/{args.doc}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if args.resume:
        # Load checkpoint.
        ckpt = os.path.join(log_dir, 'checkpoints')
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(ckpt), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(ckpt, 'ckpt.pth'))
        model.load_state_dict(checkpoint['model'])
        best_mse = checkpoint['mse']
        start_epoch = checkpoint['epoch']
        # step = checkpoint['step']
        if use_ema:# and step > ema_start_step:
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

            step = train(model, device, trainloader, optimizer, extrap=config.data.extrap, step=step)
            
            if step > ema_start_step:
                eval_model = ema_model or model
            else:
                eval_model = model
            if epoch % config.optim.eval_iter == 0 and epoch > 0:
                mse, best_mse = eval(model , device, epoch, valloader, best_mse,log_dir=log_dir,
                                         classify=config.data.classify and config.data.classify_type,
                                         extrap=config.data.extrap)
                p = 'Epoch: %d | Val mse: %1.3f' % (epoch, mse)
                if not args.debug:
                    print(p)
                else:
                    pbar.set_description(p)
            

                if mse < best_mse:
                    ckpt = os.path.join(log_dir, 'checkpoints')
                    state = {
                        'model': model.state_dict(),
                        'mse': mse,
                        'epoch': epoch,
                        'step': step
                    }
                    if ema_model and step > ema_start_step:
                        state['ema_model'] = ema_model.state_dict()
                    if not os.path.isdir(ckpt):
                        os.mkdir(ckpt)
                    torch.save(state, os.path.join(ckpt, 'ckpt.pth'))
                    best_mse = mse
            scheduler.step()
    if step > ema_start_step:
        eval_model = ema_model or model
    else:
        eval_model = model
    eval(eval_model , device, -1, valloader, best_mse,  log_dir=log_dir,
         classify=config.data.classify and config.data.classify_type,
         extrap=config.data.extrap)

if __name__ == '__main__':
    args = get_parse_args()
    args = args.parse_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    main(args)