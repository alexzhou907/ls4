import torch
import torch.nn as nn
from .s4models import S4Model, Model
import numpy as np

import wandb

class Decoder(nn.Module):
    def __init__(self, config, sigma, z_dim, in_channels, bidirectional):
        super().__init__()
        self.sigma = sigma
        self.z_dim = z_dim
        self.channels = in_channels
        self.use_spatial = config.use_spatial
        self.config = config

        self.latent = Model(
            **config.prior
        )
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.latent_back = Model(
                **config.prior
            )
            config.decoder.d_input = config.decoder.d_input * 2

        self.dec = Model(
            **config.decoder
        )


        act_type = getattr(config, 'activation', 'identity')
        if act_type == 'identity':
            self.act = nn.Identity()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'shift':
            self.act = lambda x: torch.tanh(x) + 1
        elif act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError


        self.z_prior = nn.Parameter(torch.zeros(self.z_dim))

        self.x_prior = nn.Parameter(torch.zeros(self.channels))
        if self.bidirectional:
            self.z_prior_back = nn.Parameter(torch.zeros(self.z_dim))
            self.x_prior_back = nn.Parameter(torch.zeros(self.channels))


    def sample(self, bs, t_vec,  device='cuda'):
        ''' t_vec: (seq_len+1,) '''
        if isinstance(t_vec, int):
            seq_len = t_vec
        else:
            seq_len = len(t_vec) - 1
        # first z is all 0's for start token
        zs = [self.z_prior[None].expand(bs,-1)]

        # xs = [self.x_prior[None].expand(bs,-1)]

        hidden_state_z = self.latent.default_state(bs, device=device)
        if self.bidirectional:
            zs_back = [self.z_prior_back[None].expand(bs,-1)]
            hidden_state_z_back = self.latent_back.default_state(bs, device=device)
        # hidden_state_x = self.dec.default_state(bs, device=device)
        for t in range(seq_len):
            z_t = zs[-1] #(B H)
            # xs_curr = xs[-1] #(B,C)

            # prior
            z_t, hidden_state_z = self.latent.step(z_t, None, t=None if isinstance(t_vec, int) else t_vec[t] ,state=hidden_state_z, sample=True)


            zs.append(z_t)

            if self.bidirectional:
                z_t_back = zs_back[-1]
                z_t_back, hidden_state_z_back = self.latent_back.step(z_t_back, None, t=None if isinstance(t_vec, int) else t_vec[seq_len-t-1], state=hidden_state_z_back)

                zs_back.append(z_t_back)

        zs = torch.stack(zs[1:], dim=1)

        if self.bidirectional:
            zs_back = torch.stack(zs_back[1:], dim=1).flip(1)
            # x = self.dec(zs, zs_back)
            zs = torch.cat([zs, zs_back],dim=-1)

        if self.config.decoder.aux_channels == 0:
            x = self.dec(zs, None)
        else:
            xs = [self.x_prior[None].expand(bs, -1)]
            hidden_state_x = self.dec.default_state(bs, device=device)
            for t in range(seq_len):
                x_t = xs[-1]  # (B H)

                x_t, hidden_state_x = self.dec.step(zs[:, t], x_t, t=None if isinstance(t_vec, int) else t_vec[t], state=hidden_state_x)

                xs.append(x_t)
            x = torch.stack(xs[1:], dim=1)
        x = self.act(x)
        return  x


    def extrapolate(self,x_given, t_vec, t_vec_pred, z_post):
        ''' t_vec: (seq_len+1,) '''
        device = t_vec_pred.device
        bs = z_post.shape[0]
        assert not self.bidirectional

        # xs = [self.x_prior[None].expand(bs,-1)]
        z_prior = torch.cat([self.z_prior[None,None].expand(bs,-1,-1),
                             z_post[:,:-1]], dim=1)
        hidden_state_z = self.latent.default_state(bs, device=device)


        ## get hidden state of prior
        for t in range(z_prior.shape[1]):

            # prior
            _,_,_, hidden_state_z = self.latent.step(z_prior[:,t], None, t= t_vec[t] ,state=hidden_state_z, sample=False)


        zs = [z_post[:,-1]]
        zs_mean = []
        zs_std = []

        ## extrapolate
        for t in range(len(t_vec_pred)):
            z_t = zs[-1]  # (B H)

            # prior
            z_t, z_t_mean, z_t_std, hidden_state_z = self.latent.step(z_t, None, t=t_vec_pred[t], state=hidden_state_z, sample=False)

            zs.append(z_t)
            zs_mean.append(z_t_mean)
            zs_std.append(z_t_std)


        zs = torch.stack(zs[1:], dim=1)
        zs_mean = torch.stack(zs_mean, dim=1)
        zs_std = torch.stack(zs_std, dim=1)

        zs = torch.cat([z_post, zs], dim=1)
        
        t = torch.cat([t_vec, t_vec_pred], dim=0)
        assert zs.shape[1] == len(t)
        
        if self.config.decoder.aux_channels == 0:
            x = self.dec(zs, None, t=t)
            x = self.act(x)
            
            pred_len = len(t_vec_pred)
            x = x[:, -pred_len:]
        else:
            xs = torch.cat([self.x_prior[None,None].expand(bs,-1, -1),
                            x_given[:,:-1]],dim=1)
            hidden_state_x = self.dec.default_state(bs, device=device)

            for t in range(len(t_vec)):
                _, hidden_state_x = self.dec.step(zs[:, t], xs[:, t], t=t_vec[t], state=hidden_state_x, sample=True)


            xs = [x_given[:,-1]]
            for t in range(len(t_vec_pred)):
                x_t = xs[-1]  # (B H)

                x_t, hidden_state_x = self.dec.step(zs[:,len(t_vec)+ t],
                                                    x_t, t=t_vec_pred[t],
                                                    state=hidden_state_x, sample=True)

                xs.append(self.act(x_t))
            x = torch.stack(xs[1:], dim=1)

        return  x, self.sigma * torch.ones_like(x), zs_mean, zs_std

    def reconstruct(self, x, t_vec, z_post, z_post_back=None): # decoder.reconstruct
        # B, L = z_post.shape[:2]

        x_input = torch.cat([self.x_prior[None, None].expand(x.shape[0], -1, -1), x[:, :-1]], dim=1)

        if self.bidirectional and z_post_back is not None:
            # x_input_back = torch.cat([self.x_prior_back[None, None].expand(x.shape[0], -1, -1), x.flip(1)[:, :-1]], dim=1)
            z_post_back = z_post_back.flip(1)
            x, x_std = self.decode(torch.cat([z_post, z_post_back],dim=-1),
                                            x_input, t_vec)
            # x, _ = self.decode(z_post, z_post_back)
        else:

            x, x_std = self.decode(z_post, x_input, t_vec)

        return x, x_std

    def decode(self, z, x, t_vec):
        """ z: (b, l, z_dim) """
        x = self.act(self.dec(z, x, t=t_vec))

        return x, self.sigma * torch.ones_like(x)

    def forward(self, x, t_vec, z_post, z_post_back=None):
        '''z_post: (B, L, z_dim)'''

        z_input = torch.cat([
            self.z_prior[None,None].expand(z_post.shape[0],-1,-1),
            z_post[:,:-1]
        ], dim=1)

        x_input = torch.cat([self.x_prior[None,None].expand(x.shape[0],-1,-1), x[:,:-1]], dim=1)
        # x_input = None
        prior_mean, prior_std = self.latent(z_input, x_input, t=t_vec)
        if self.bidirectional and z_post_back is not None:
            z_input_back = torch.cat([
                self.z_prior_back[None,None].expand(z_post_back.shape[0],-1,-1),
                z_post_back[:,:-1]
            ], dim=1)
            x_input_back = torch.cat([self.x_prior_back[None,None].expand(x.shape[0],-1,-1), x.flip(1)[:,:-1]], dim=1)
            prior_mean_back, prior_std_back = self.latent_back(z_input_back, x_input_back, t=t_vec.flip(-1))

            z_post_back_flip =  z_post_back.flip(1)
            dec_mean, dec_std = self.decode(torch.cat([z_post, z_post_back_flip],dim=-1),
                                            x_input, t_vec)
            # dec_mean, dec_std = self.decode(z_post,
            #                                 z_post_back_flip,)

            return dec_mean, dec_std, prior_mean, prior_std, prior_mean_back, prior_std_back
        else:

            dec_mean, dec_std = self.decode(z_post, x_input, t_vec)
            return dec_mean, dec_std, prior_mean, prior_std


class Encoder(nn.Module):
    def __init__(self, config, z_dim, bidirectional):
        super().__init__()

        self.z_dim = z_dim
        self.use_spatial = config.use_spatial
        self.bidirectional = bidirectional
        self.latent = Model(
            **config.posterior
        )

        if self.bidirectional:
            self.latent_back = Model(
                **config.posterior
            )

    def _reparameterized_sample(self, mu, std):

        eps = torch.empty(size=std.size(), device=mu.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mu)

    def encode(self, x, t_vec, use_forward=True):
        if use_forward:
            post_mean, post_std = self.latent(x, t=t_vec)
        else:
            assert self.bidirectional

            post_mean, post_std = self.latent_back(x.flip(1), t=t_vec.flip(-1))

        z = self._reparameterized_sample(post_mean, post_std)

        return z, post_mean, post_std



class VAE(nn.Module):
    EPS =  torch.finfo(torch.float).eps
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config.encoder, config.z_dim, config.bidirectional)

        self.decoder = Decoder(config.decoder, config.sigma, config.z_dim, config.in_channels, config.bidirectional)

        self.config = config
        self.bidirectional = config.bidirectional

        self.use_classifier = config.get('classifier', False)
        self.n_labels = config.n_labels
        n_input = config.z_dim * 2 if self.bidirectional else config.z_dim
        if self.use_classifier:
            self.clf_type = config.get('classifier_type', 'per_seq')
            if config.linear_cls:
                self.classifier = nn.Linear(n_input, config.n_labels)
            else:

                self.classifier = nn.Sequential(
                    nn.Linear(n_input, 300),
                    nn.ReLU(),
                    nn.Linear(300, 300),
                    nn.ReLU(),
                    nn.Linear(300, config.n_labels), )

    def forward(self, x, timepoints, masks, labels=None, plot=False , sum=False):

        # B,L = x.shape[:2]
        log_info = {}
        z_post, z_post_mean, z_post_std = self.encoder.encode(x, timepoints, use_forward=True) # (B, L , z_dim)

        if self.bidirectional:

            z_post_back, z_post_mean_back, z_post_std_back = self.encoder.encode(x, timepoints, use_forward=False)  # (B, L , z_dim)

            dec_mean, dec_std, z_prior_mean, z_prior_std, z_prior_mean_back, z_prior_std_back = self.decoder(x,timepoints, z_post, z_post_back)  # (B, L, z_dim)

            kld_loss_forward = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=sum)  # (B,)
            kld_loss_back = self._kld_gauss(z_post_mean_back, z_post_std_back,
                                       z_prior_mean_back, z_prior_std_back, masks, sum=sum)  # (B,)

            kld_loss = kld_loss_forward + kld_loss_back

            log_info.update({'z_post_mean_forward': z_post_mean.mean().detach().cpu().item(),
                              'z_post_std_forward': z_post_std.mean().detach().cpu().item(),
                              'z_post_mean_back' :z_post_mean_back.mean().detach().cpu().item(),
                              'z_post_std_back': z_post_std_back.mean().detach().cpu().item(),
                              'z_prior_mean_forward': z_prior_mean.mean().detach().cpu().item(),
                              'z_prior_std_forward': z_prior_std.mean().detach().cpu().item(),
                              'z_prior_mean_back': z_prior_mean_back.mean().detach().cpu().item(),
                              'z_prior_std_back': z_prior_std_back.mean().detach().cpu().item(),
                              })
        else:

            dec_mean, dec_std, z_prior_mean, z_prior_std = self.decoder(x, timepoints, z_post) # (B, L, z_dim)
            
            kld_loss = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=sum) # (B,)

            log_info.update({
                'z_post_mean_forward': z_prior_mean.mean().detach().cpu().item(),
                'z_post_std_forward': z_post_std.mean().detach().cpu().item(),
                'z_prior_mean_forward': z_prior_mean.mean().detach().cpu().item(),
                'z_prior_std_forward': z_prior_std.mean().detach().cpu().item(),
            })

        if plot:
            full_nll = (
                self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=True) +\
                self._nll_gauss(dec_mean, dec_std, x, masks, sum=True)
            )
            full_nll = full_nll.mean()
            log_info.update({'full_nll': full_nll.detach().cpu().item()})


        nll_loss =  self._nll_gauss(dec_mean, dec_std, x, masks, sum=sum)

        loss = (kld_loss + nll_loss).mean()

        if labels is not None and self.use_classifier:
            if self.bidirectional:
                z_post = torch.cat([z_post, z_post_back.flip(1)], dim=-1)
            if self.clf_type == 'per_seq':
                masks = (masks.sum(-1) > 0)[...,None]
                masks_sum = masks.sum(1)
                masks_sum[masks_sum == 0] = 1.
                pred = self.classifier((z_post * masks).sum(1)/ masks_sum )
            elif self.clf_type == 'per_tp':

                pred = self.classifier(z_post)
            else:
                raise NotImplementedError
            if self.n_labels  == 1:
                ce_loss = binary_ce_loss(pred, labels)
            else:
                ce_loss = multiclass_ce_loss(pred,labels, masks)

            loss = loss + 100*ce_loss
            log_info.update({'ce_loss': ce_loss.detach().cpu().item()})
        log_info.update({
                  'dec_mean':dec_mean.mean().detach().cpu().item(),
                  'kld_loss':kld_loss.mean().detach().cpu().item(),
                  'nll_loss':nll_loss.mean().detach().cpu().item(),
                'mse_loss': self._mse(dec_mean, x, masks, sum=False).mean().detach().cpu().item(),
                       'loss': loss.detach().cpu().item(),})


        return loss, log_info


    def setup_rnn(self, mode='dense'):
        """
        Convert the SaShiMi model to a RNN for autoregressive generation.
        Args:
            mode: S4 recurrence mode. Using `diagonal` can speed up generation by 10-20%.
                `linear` should be faster theoretically but is slow in practice since it
                dispatches more operations (could benefit from fused operations).
                Note that `diagonal` could potentially be unstable if the diagonalization is numerically unstable
                (although we haven't encountered this case in practice), while `dense` should always be stable.
        """
        assert mode in ['dense', 'diagonal', 'linear']
        for module in self.modules():
            if hasattr(module, 'setup_step'): module.setup_step(mode=mode)


    def generate(self, B,  timepoints, device='cuda'):


        x = self.decoder.sample(B, timepoints, device=device)

        return x

    def predict(self, x, t, masks):

        assert self.use_classifier
        z, z_mean, z_std = self.encoder.encode(x, t)

        if self.bidirectional:
            z_back, z_back_mean, _ = self.encoder.encode(x, t, use_forward=False)
            z = torch.cat([z, z_back.flip(1)], dim=-1)
        if self.clf_type == 'per_seq':

            masks = (masks.sum(-1) > 0)[..., None]
            masks_sum = masks.sum(1)
            masks_sum[masks_sum == 0] = 1.
            pred = self.classifier((z * masks).sum(1) / masks_sum)

        elif self.clf_type == 'per_tp':

            pred = self.classifier(z)

        else:
            raise NotImplementedError
        _, pred = pred.max(-1)
        return pred

    def reconstruct(self, x, t_vec, t_vec_pred=None, x_full=None, masks=None, get_full_nll=False):
        '''if t_vec_pred is None: recon  x, if not, predict t_vec_pred's x '''

        z, z_mean, z_std = self.encoder.encode(x, t_vec, use_forward=True)
        if self.bidirectional:
            z_post_back, z_post_mean_back, z_post_std_back = self.encoder.encode(x, t_vec,
                                                                                 use_forward=False)  # (B, L , z_dim)
        else:
            z_post_back = None

        if t_vec_pred is None:

            x, _ = self.decoder.reconstruct(x, t_vec, z, z_post_back)  # (B, L, z_dim)
            assert not get_full_nll
        else:

            dec_mean, dec_std, z_prior_mean, z_prior_std = self.decoder.extrapolate(x, t_vec, t_vec_pred, z)  # (B, L, z_dim)

            if get_full_nll:
                t_full = torch.cat([t_vec, t_vec_pred], dim=0)
                _, z_mean_full, z_std_full = self.encoder.encode(x_full, t_full, use_forward=True)
                z_mean_pred = z_mean_full[:, -len(t_vec_pred):]
                z_std_full = z_std_full[:, -len(t_vec_pred):]
                kld_loss = self._kld_gauss(z_mean_pred, z_std_full, z_prior_mean, z_prior_std, masks, sum=True)  # (B,)

                full_nll = (
                        kld_loss + self._nll_gauss(dec_mean, dec_std, x_full[:,-len(t_vec_pred):], masks, sum=True)
                )
                full_nll = full_nll.mean()
                return dec_mean, full_nll

            x = dec_mean
        return x

    def _get_alpha_ts(self, schedule='linear', alpha_min=0, alpha_max=1, ts=40):
        alphas = None
        if schedule == 'linear':
            alphas = torch.linspace(alpha_min, alpha_max, ts)
        elif schedule == 'quad':
            alphas = torch.linspace(alpha_min ** 0.5, alpha_max ** 0.5, ts) ** 2
        return alphas


    def interpolate(self, x1, x2, ts, device, t_vec=None, t_vec_pred=None, x_full=None, get_full_nll=False):
        '''if t_vec_pred is None: recon  x, if not, predict t_vec_pred's x 
        Interpolate between different timesteps per two seqqueces of poses.'''
        print(f'x1: {x1.shape}')
        
        z1, z1_mean, z1_std = self.encoder.encode(x1, t_vec, use_forward=True)
        z2, z2_mean, z2_std = self.encoder.encode(x2, t_vec, use_forward=True) # (bs, T, z_dim)
        z_post_back = None
        assert not self.bidirectional, f'not supposed to be bidirectional'

        # do mixing / interpolation in latent
        alpha_ts = self._get_alpha_ts(self.config.interp_schedule, self.config.alpha_min, self.config.alpha_max, ts)
        alpha_ts = alpha_ts[None, ..., None]
        alpha_ts = alpha_ts.to(device)
        
        print(f'z1 {z1.shape} z2 {z2.shape} | alphas {alpha_ts.shape}')
        z_new = alpha_ts * z1 + (1 - alpha_ts) * z2

        x, _ = self.decoder.reconstruct(x1, t_vec, z_new, z_post_back)  # (B, L, z_dim)
        assert not get_full_nll

        return x


    def encode(self, x, t_vec): # get z's, i.e. the samples

        z, z_mean, z_std = self.encoder.encode(x, t_vec, use_forward=True)

        return z, z_mean, z_std

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2, masks=None, sum=False):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + self.EPS) - 2 * torch.log(std_1 + self.EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        if masks is None:
            if sum:
                loss = 0.5 * torch.sum(kld_element.flatten(1), dim=1)
            else:
                loss = 0.5 * torch.mean(kld_element.flatten(1), dim=1)
        else:
            masks =(masks.sum(-1)>0)
            masks_sum = masks.sum(-1)
            masks_sum[masks_sum==0] = 1
            if sum:
                loss = 0.5 * torch.sum((kld_element * masks[..., None]).sum(-1), dim=-1)
            else:
                loss =	0.5 * torch.sum((kld_element * masks[...,None]).mean(-1), dim=-1) / masks_sum
        return loss

    def _nll_gauss(self, mean, std, x, masks, sum=False):
        # (B,L,z_dim)
        nll = (torch.log(std + self.EPS) + np.log(2*np.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
        masks_sum = masks.sum(1)
        masks_sum[masks_sum==0] = 1.

        if sum:
            loss = torch.sum((nll * masks).sum(1), dim=-1)
        else:
            loss =torch.sum((nll * masks).sum(1) / masks_sum, dim=-1)
        return loss


    def _mse(self, mean, x, masks, sum=False):
        # (B,L,z_dim)
        nll = (x - mean).pow(2)
        masks_sum = masks.sum(1)
        masks_sum[masks_sum==0] = 1.

        if sum:
            loss = torch.sum((nll * masks).sum(1), dim=-1)
        else:
            loss =torch.mean((nll * masks).sum(1) / masks_sum, dim=-1)
        return loss

    def full_nll(self, x, timepoints, masks=None):

        z_post, z_post_mean, z_post_std = self.encoder.encode(x, timepoints, use_forward=True)  # (B, L , z_dim)

        if self.bidirectional:
            z_post_back, z_post_mean_back, z_post_std_back = self.encoder.encode(x, timepoints,
                                                                                 use_forward=False)  # (B, L , z_dim)

            dec_mean, dec_std, z_prior_mean, z_prior_std, z_prior_mean_back, z_prior_std_back = self.decoder(x,
                                                                                                             timepoints,
                                                                                                             z_post,
                                                                                                             z_post_back)  # (B, L, z_dim)

            kld_loss_forward = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=True)  # (B,)
            kld_loss_back = self._kld_gauss(z_post_mean_back, z_post_std_back,
                                            z_prior_mean_back, z_prior_std_back, masks, sum=True)  # (B,)

            kld_loss = kld_loss_forward + kld_loss_back
        else:

            dec_mean, dec_std, z_prior_mean, z_prior_std = self.decoder(x, timepoints, z_post) # (B, L, z_dim)

            kld_loss = self._kld_gauss(z_post_mean, z_post_std, z_prior_mean, z_prior_std, masks, sum=True) # (B,)

        full_nll = (
                kld_loss+ self._nll_gauss(dec_mean, dec_std, x, masks, sum=True)
        )
        full_nll = full_nll.mean()
        
        return full_nll

def binary_ce_loss(pred, labels):
    labels = labels.reshape(-1)
    pred = pred.reshape(-1)
    idx_not_nan = ~torch.isnan(labels)
    if len(idx_not_nan) == 0.:
        print("All are labels are NaNs!")
        ce_loss = torch.Tensor(0.).to(labels.device)
        return ce_loss
    labels = labels[idx_not_nan]
    pred = pred[idx_not_nan]

    assert (not torch.isnan(pred).any())
    assert (not torch.isnan(labels).any())

    if torch.sum(labels == 0.) == 0 or torch.sum(labels == 1.) == 0:
        print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

    ce_loss = nn.BCEWithLogitsLoss(pos_weight=(labels==0).sum()/(labels==1).sum() )(pred, labels)

    return ce_loss


def multiclass_ce_loss(pred, labels, mask):
    pred = pred.reshape(-1, pred.shape[-1])
    labels = labels.reshape(-1, labels.shape[-1])
    mask = (torch.sum(mask, -1) > 0).reshape(-1)
    # pred_mask = mask[...,None].repeat(1,1,pred.shape[-1])
    # label_mask = mask[...,None]
    if (pred.size(-1) > 1) and (labels.size(-1) > 1):
        assert (pred.size(-1) == labels.size(-1))
        # targets are in one-hot encoding -- convert to indices
        _, labels = labels.max(-1)
    labels = labels.squeeze(-1)

    ce_loss = nn.CrossEntropyLoss(reduction='none')(pred, labels.long())

    ce_loss = torch.masked_select(ce_loss, mask)

    ce_loss = ce_loss.mean()

    return ce_loss
