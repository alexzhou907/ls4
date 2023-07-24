from importlib.metadata import requires
import logging

import torch
import torch.nn as nn
from .s4 import S4
from .s4d import S4D, S4DJoint, Activation

from .seq_unet import SequentialUnet, s4_registry

if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d
import math
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_state,
        # d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        bidirectional=False,
        s4_type='s4',
        use_aux=False,
        lr=0.001
    ):
        super().__init__()

        self.prenorm = prenorm
        self.use_aux = use_aux

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                s4_registry[s4_type](
                    d_model=d_model,
                    d_state=d_state,
                    bidirectional=bidirectional,
                    # postact='glu' if glu else None,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, lr)
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout) if dropout > 0 else nn.Identity())

        # Linear decoder
        # self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, aux=None, t=None, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            if self.use_aux and aux is not None:
                (z, aux), _ = layer((z, aux))

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        # x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)

        return x, None

    def default_state(self, *args, **kwargs):
        return [layer.default_state(*args, **kwargs) for layer in self.s4_layers]

    def step(self, x, state: list , t=None):
        '''
        x: (B, H)
        state: (B, H, N)
        return: (B,  d_output)
        '''
        state = state[::-1]
        next_state = []
        x = self.encoder(x)

        for layer, norm in zip(self.s4_layers, self.norms):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, next_state_ = layer.step(z, state=state.pop())
            next_state.append(next_state_)
            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        # x = self.linear(x)
        # x = self.act(x)

        return x, next_state



class Model(nn.Module):
    
    def __init__(
        self,
        d_input,
        aux_channels,
        d_state,
        d_output=10,
        d_model=256,
        d_temb=0,
        n_inheads=1,
        n_outheads=1,
        n_auxinheads=1,
        n_layers=4,
        bidirectional=False,
        dropout=0.2,
        s4_type='s4',
        mix_aux=True,
        backbone='autoreg',
        use_unet=True,
        #unet params
        pool=[4,4],
        expand=2,
        ff=2,
        #latent
        use_latent=False,
        latent_type='none',
        aux_out=0,
        lr=0.001
    ):
        super().__init__()
        self.bidirectional = bidirectional
        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.use_aux = (aux_channels != 0)
        self.mix_aux = mix_aux
        self.s4_type = s4_type
        self.d_temb = d_temb
        self.n_inheads = n_inheads
        self.n_outheads = n_outheads
        self.n_auxinheads = n_auxinheads
        if self.d_temb > 0:
            self.temb = nn.Sequential(
                torch.nn.Linear(self.d_temb*2,
                                self.d_temb*4),
                torch.nn.SiLU(),
                torch.nn.Linear(self.d_temb*4,
                                self.d_temb),
            )
        self.encoder = nn.Linear(d_input*self.n_inheads, d_model)
        if self.s4_type == 's4d_joint' and self.use_aux:
            assert not bidirectional
            self.aux_encoder_in = nn.Linear(aux_channels*self.n_auxinheads, d_model)
        elif self.mix_aux and self.use_aux:

            self.aux_encoder_in = nn.Linear(aux_channels*self.n_auxinheads, d_model)
        elif self.use_aux:
            self.aux_encoder_in = nn.Linear(aux_channels*self.n_auxinheads, d_model)

            self.comb = nn.GLU()
        # Stack S4 layers as residual blocks
        if backbone == 'autoreg':
            
            self.backbone = SequentialUnet(
                d_model=d_model,
                d_state=d_state,
                d_temb=d_temb,
                n_layers=n_layers,
                pool=pool,
                expand=expand,
                ff=ff,
                bidirectional=bidirectional,
                unet=use_unet,
                s4_type=s4_type,
                mix_aux=mix_aux,
                dropout=dropout,
                use_aux=self.use_aux,
            )
        elif backbone == 'seq':
            self.backbone = S4Model(d_input=d_model,
                                    d_state=d_state,
                                    d_model=d_model,
                                    n_layers=n_layers,
                                    dropout=dropout,
                                    bidirectional=bidirectional,
                                    s4_type=s4_type,
                                    use_aux=self.use_aux and s4_type == 's4d_joint',
                                    lr=lr)
        else:
            raise NotImplementedError
        self.use_latent = use_latent
        self.use_multihead = aux_out != 0

        self.latent_type = latent_type


        if self.use_multihead:
            self.aux_linear = nn.Linear(d_model, aux_out)

        if use_latent:

            if latent_type == 'none':
                self.linear = nn.Linear(d_model, d_output)
                self.latent = S5D(d_model=d_output, d_state=d_state,
                                  transposed=True,
                                  lr=min(0.001, lr))
            elif latent_type == 'split':
                self.linear = nn.Linear(d_model, d_output)
                self.mean = s4_registry[s4_type](d_model=d_output, d_state=d_state,
                                  transposed=True,
                                  lr=min(0.001, lr))
                self.log_std = s4_registry[s4_type](d_model=d_output, d_state=d_state,
                                  transposed=True,
                                  lr=min(0.001, lr))
            elif latent_type == 'const_std':
                self.linear = nn.Linear(d_model, d_output)
                self.mean = s4_registry[s4_type](d_model=d_output, d_state=d_state,
                                  transposed=True,
                                  lr=min(0.001, lr))

                self.log_std = nn.Parameter(torch.ones(d_output) * -5)

            elif latent_type == 'single':

                self.linear = nn.Linear(d_model, d_output)
                if self.use_aux:
                    if self.s4_type == 's4d_joint':

                        self.aux_encoder = nn.Linear(d_model, d_output)
                    else:

                        self.aux_encoder = nn.Linear(aux_channels, d_output)

                    self.comb = nn.GLU()
                self.latent = S5D(d_model=d_output, d_state=d_state,
                                        transposed=True,
                                        lr=min(0.001, lr))
            elif latent_type == 'joint':
                assert self.use_aux, 'joint type latent state must use auxiliary input'

                self.linear = nn.Linear(d_model, d_output)
                if self.s4_type == 's4d_joint':

                    self.aux_encoder = nn.Linear(d_model, d_output)
                else:

                    self.aux_encoder = nn.Linear(aux_channels, d_output)

                self.latent = S5DJoint(d_model=d_output, d_state=d_state,
                                  transposed=True,
                                  lr=min(0.001, lr))
            else:
                raise ValueError('Latent type not supported')
        else:
            if self.n_outheads > 1:
                self.linear = nn.ModuleList([nn.Linear(d_model, d_output) for _ in range(self.n_outheads)])
            else:
                self.linear = nn.Linear(d_model, d_output)


    def forward(self, x, aux=None, t=None):
        """
        Input x, aux is shape  (B, L, d_input), (B, L, aux_channel)
        t : (L,) time evaluated
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        if self.d_temb > 0 and t is not None:
            t = get_timestep_embedding(t, self.d_temb*2)
            t = self.temb(t)
        else:
            t = None
        if aux is not None and self.use_aux and (self.s4_type == 's4d_joint' or self.mix_aux):
            aux = self.aux_encoder_in(aux)
            (x, aux), _ = self.backbone((x, aux), t=t)

        elif aux is not None and self.use_aux:
            aux = self.aux_encoder_in(aux)
            x = torch.cat([x, aux], dim=-1)
            x = self.comb(x)
            x, _ = self.backbone(x, t=t)
        else:
            x, _ = self.backbone(x, t=t)

        if self.use_multihead:
            aux_out = self.aux_linear(x)

        if self.use_latent:
            if self.latent_type == 'single':
                assert self.use_aux
                # Decode the outputs
                x = self.linear(x)
                aux = self.aux_encoder(aux)
                x = torch.cat([x, aux], dim=-1)
                x = self.comb(x)
                x = x.transpose(-1, -2)

                x_mean, x_std = self.latent(x, t=t)  # (B, L, d_model) -> (B, L, z_dim)

                x_mean = x_mean.transpose(-1, -2)
                x_std = x_std.transpose(-1, -2)
            elif self.latent_type == 'joint':
                assert self.use_aux
                # Decode the outputs
                x = self.linear(x)
                aux = self.aux_encoder(aux)

                x = x.transpose(-1, -2)
                aux = aux.transpose(-1, -2)
                x = (x, aux)

                x_mean, x_std = self.latent(x, t=t)  # (B, L, d_model) -> (B, L, z_dim)

                x_mean = x_mean.transpose(-1, -2)
                x_std = x_std.transpose(-1, -2)
            elif self.latent_type == 'split':
                x = self.linear(x)
                x = x.transpose(-1, -2)

                x_mean, _ = self.mean(x, t=t)
                x_log_std, _ = self.log_std(x, t=t)
                x_std = torch.exp(x_log_std)


                x_mean = x_mean.transpose(-1, -2)
                x_std = x_std.transpose(-1, -2)
            elif self.latent_type == 'const_std':
                x = self.linear(x)
                x = x.transpose(-1, -2)

                x_mean, _ = self.mean(x, t=t)
                x_std = torch.exp(self.log_std)[None, :, None].expand_as(x_mean)


                x_mean = x_mean.transpose(-1, -2)
                x_std = x_std.transpose(-1, -2)

            elif self.latent_type == 'none':

                # Decode the outputs
                x = self.linear(x)
                x = x.transpose(-1, -2)

                x_mean, x_std = self.latent(x, t=t)  # (B, L, d_model) -> (B, L, z_dim)

                x_mean = x_mean.transpose(-1, -2)
                x_std = x_std.transpose(-1, -2)
            else:
                raise NotImplementedError

            if self.use_multihead:
                return x_mean, x_std, aux_out
            else:
                return x_mean, x_std
        else:

            # Decode the outputs
            if self.n_outheads > 1:
                x = torch.stack([lin(x) for lin in self.linear], dim=-2)
            else:
                x = self.linear(x)
            if self.use_multihead:
                return x, aux_out
            else:
                return x

    def default_state(self, *batch_shape, device='cuda'):
        unet_state = self.backbone.default_state(*batch_shape, device=device)

        if self.use_latent and self.latent_type == 'split':
            latent_default_mean = self.mean.default_state(*batch_shape, device=device)
            latent_default_std = self.log_std.default_state(*batch_shape, device=device)

            return {'unet': unet_state, 'latent': (latent_default_mean, latent_default_std)}
        elif self.use_latent and self.latent_type == 'const_std':
            latent_default = self.mean.default_state(*batch_shape, device=device)

            return {'unet': unet_state, 'latent': latent_default}
        elif self.use_latent and self.latent_type != 'split':
            latent_default = self.latent.default_state(*batch_shape, device=device)

            return {'unet': unet_state, 'latent': latent_default}
        else:
            return {'unet': unet_state}

    def step(self, x, aux=None, state: dict=None, t=None, sample=False):
        '''
        x: (B, H)
        state: (B, H, N)
        return: (B,  d_output)
        t: 0-d tensor
        '''
        if aux is None: assert not self.use_aux
        next_state = {}
        x = self.encoder(x)
        if self.d_temb > 0 and t is not None:
            t = get_timestep_embedding(t[None], self.d_temb*2)
            t = self.temb(t)
        if aux is not None and self.use_aux and (self.s4_type == 's4d_joint' or self.mix_aux):
            aux = self.aux_encoder_in(aux)
            (x, aux), unet_next_state = self.backbone.step((x, aux), state=state['unet'], t=t)
        elif aux is not None and self.use_aux:
            aux = self.aux_encoder_in(aux)
            x = torch.cat([x, aux], dim=-1)
            x = self.comb(x)
            x, unet_next_state = self.backbone.step(x, state=state['unet'], t=t)
        else:
            x, unet_next_state = self.backbone.step(x, state=state['unet'], t=t)

        # breakpoint()
        next_state['unet'] = unet_next_state

        if self.use_multihead:
            aux_out = self.aux_linear(x)

        if self.use_latent:

            if self.latent_type == 'single':
                x = self.linear(x)
                assert self.use_aux
                aux = self.aux_encoder(aux)
                x = torch.cat([x, aux], dim=-1)
                x = self.comb(x)

                x, next_state_ = self.latent.step(x, state=state['latent'], t=t,  sample=True)
                next_state['latent'] = next_state_
            elif self.latent_type == 'split':

                x = self.linear(x)

                x_mean, next_state_mean = self.mean.step(x, state=state['latent'][0])
                x_log_std, next_state_std = self.log_std.step(x, state=state['latent'][1])

                next_state['latent'] = (next_state_mean, next_state_std)
                x_std = torch.exp(x_log_std)

                eps = torch.empty(size=x_std.size(), device=x_std.device, dtype=torch.float).normal_()

                x = eps.mul(x_std).add_(x_mean) # sample
            elif self.latent_type == 'const_std':
                x = self.linear(x)

                x_mean, next_state_mean = self.mean.step(x, state=state['latent'])
                x_std = torch.exp(self.log_std)[None].expand_as(x_mean)

                next_state['latent'] = next_state_mean
                
                eps = torch.empty(size=x_std.size(), device=x_std.device, dtype=torch.float).normal_()

                x = eps.mul(x_std).add_(x_mean) # sample
            elif self.latent_type == 'joint':
                assert self.use_aux
                x = self.linear(x)
                aux = self.aux_encoder(aux)

                x = (x, aux)

                x, next_state_ = self.latent.step(x, state=state['latent'], t=t,  sample=True)
                next_state['latent'] = next_state_
            elif self.latent_type == 'none':
                # Decode the outputs
                x = self.linear(x)
                x, next_state_ = self.latent.step(x, state=state['latent'], t=t, sample=True)
                next_state['latent'] = next_state_

            if self.use_multihead:
                if sample:
                    return x,  aux_out, next_state
                else:
                    return x, x_mean, x_std, aux_out, next_state
            else:
                if sample:
                    return x, next_state
                else:
                    return x, x_mean, x_std, next_state
        else:
            if self.n_outheads > 1:
                x = torch.stack([lin(x) for lin in self.linear], dim=-2)
            else:
                x = self.linear(x)
            if self.use_multihead:
                return x, aux_out, next_state
            else:
                return x, next_state



def test_step():
    device = 'cuda'
    B = 2
    H = 1
    N = 4
    L = 784
    # s4 = LatentS4Model(H, d_state=N, d_output=H, d_model=10, n_layers=2, dropout=0, transposed=True )
    s4 = Model(d_input=H, aux_channels=H, d_state=N, d_output=3, d_model=10, n_layers=1,
                     dropout=0, s4_type='s4d', mix_aux=True,use_latent=True, latent_type='split')
    s4.to(device)
    # s4.eval()

    for module in s4.modules():
        if hasattr(module, 'setup_step'): module.setup_step()

    u = 100*torch.ones(B, L, H).to(device)
    x = 100*torch.ones(B, L, H).to(device)
    initial_state = s4.default_state(B, device=device)

    # initial_state = initial_state[.., :N // 2]
    # state = [iss.clone() for iss in initial_state]
    y, y_std = s4(u, x)
    print("output mean:\n", y, y.shape)
    print("output y_std:\n", y_std, y_std.shape)
    # print("final state:\n", final_state, final_state.shape)

    # Use Stepping

    # state =[iss.clone() for iss in initial_state]
    state = initial_state
    ys = []
    ystds = []
    for i, u_ in enumerate(torch.unbind(u, dim=1)):
        _, y_, y_std_, state = s4.step(u_, x[:,i], state=state, sample=False)
        ys.append(y_)
        ystds.append(y_std_)
    ys = torch.stack(ys, dim=1)
    ystds = torch.stack(ystds, dim=1)
    print("step outputs mean:\n", ys, ys.shape)
    print("step outputs y_aux:\n", ystds, ystds.shape)
    # print("step final state:\n", state)

    breakpoint()


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    test_step()