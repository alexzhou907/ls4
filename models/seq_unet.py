import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from opt_einsum import contract, contract_expression
from functools import partial
from .s4 import S4
from .s4d import S4D, S4DJoint


s4_registry = {'s4': S4, 's4d': S4D, 's4d_joint': S4DJoint}


class modrelu(nn.Module):
    def __init__(self, features):
        # For now we just support square layers
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude

class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)


def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer


class TransposedLinear(nn.Module):
    """ Linear module on the second-to-last dimension """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # nn.Linear default init
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias


class TransposedLN(nn.Module):
    """ LayerNorm module over second-to-last dimension
    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup
    """
    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            s, m = torch.std_mean(x, dim=-2, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            y = self.ln(x.transpose(-1,-2)).transpose(-1,-2)
        return y


def LinearActivation(
        d_input, d_output, bias=True,
        zero_bias_init=False,
        transposed=False,
        initializer=None,
        activation=None,
        activate=False, # Apply activation as part of this module
        weight_norm=False,
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear

class DownPool(nn.Module):
    def __init__(self, d_input, expand, pool, use_aux=False):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = LinearActivation(
            d_input * pool,
            self.d_output,
            transposed=True,
        )

        if use_aux:
            self.linear_aux = LinearActivation(
                d_input * pool,
                self.d_output,
                transposed=True,
            )

    def forward(self, x, t=None, **kwargs):
        has_aux = False
        if isinstance(x, tuple):
            has_aux = True
            x, aux = x
            aux = rearrange(aux, '... h (l s) -> ... (h s) l', s=self.pool)
            aux = self.linear_aux(aux)
            
        x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.pool)
        x = self.linear(x)
        if t is not None:
            t = rearrange(t, '... (l s) c -> ... l (s c)', s=self.pool)
        if has_aux:
            return (x, aux), t, None
        else:
            return x, t, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        has_aux = False
        if x is None: return None, state
        if isinstance(x, tuple):
            has_aux = True
            x, aux = x
            state.append((x, aux))
        else:
            state.append(x)
        if len(state) == self.pool:
            if has_aux:
                state, aux_state = tuple(zip(*state))
                aux = rearrange(torch.stack(aux_state, dim=-1), '... h s -> ... (h s)')
                aux = self.linear_aux(aux.unsqueeze(-1)).squeeze(-1)
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)

            if has_aux:
                return (x, aux), []
            else:
                return x, []

        else:
            return None, state

    def default_state(self, *args, **kwargs):
        return []


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, use_aux=False):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.use_aux = use_aux

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
        )

        if use_aux:
            self.linear_aux = LinearActivation(
                d_input,
                self.d_output * pool,
                transposed=True,
            )

    def forward(self, x, skip=None, t=None, **kwargs):
        has_aux=False
        if isinstance(x, tuple):
            has_aux = True
            x, aux = x
            aux = self.linear_aux(aux)
            aux = F.pad(aux[..., :-1], (1, 0))  # Shift to ensure causality
            aux = rearrange(aux, '... (h s) l -> ... h (l s)', s=self.pool)


        x = self.linear(x)

        x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip

        if t is not None:
            t = rearrange(t, '... l (s c) -> ... (l s) c', s=self.pool)

        if has_aux:
            return (x, aux), t, None
        else:
            return x, t, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0

        y, state = state[0], state[1:]
        if len(state) == 0:
            if self.use_aux:
                assert x[0] is not None
            else:
                assert x is not None
            has_aux = False
            if isinstance(x, tuple):
                has_aux = True
                x, aux = x
                aux = self.linear_aux(aux.unsqueeze(-1)).squeeze(-1)
                aux = rearrange(aux, '... (h s) -> ... h s', s=self.pool)
                aux_state = list(torch.unbind(aux, dim=-1))

            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))

            if has_aux:
                state = list(zip(state, aux_state))

        else:
            if self.use_aux:
                assert x[0] is None
            else:
                assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        if self.use_aux:
            aux_state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device)  # (batch, h, s)
            aux_state = list(torch.unbind(aux_state, dim=-1))
            state = list(zip(state, aux_state))
        return state


class FFBlock(nn.Module):

    def __init__(self, d_model, expand=2, dropout=0.0, use_aux=False):
        """
        Feed-forward block.
        Args:
            d_model: dimension of input
            expand: expansion factor for inverted bottleneck
            dropout: dropout rate
        """
        super().__init__()
        self.use_aux = use_aux
        if use_aux:
            d_model = d_model*2
        input_linear = LinearActivation(
            d_model,
            d_model * expand,
            transposed=True,
            activation='gelu',
            activate=True,
        )
        dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        output_linear = LinearActivation(
            d_model * expand,
            d_model,
            transposed=True,
            activation=None,
            activate=False,
        )

        self.ff = nn.Sequential(
            input_linear,
            dropout,
            output_linear,
        )

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            x, aux = x
            x_dim = x.shape[1]
            x = torch.cat([x, aux], dim=1)
            res = self.ff(x)
            res, res_aux = res[..., :x_dim, :], res[..., x_dim:, :]
            return (res, res_aux), None
        else:
            return self.ff(x), None

    def default_state(self, *args, **kwargs):
        return None

    def step(self, x, state, **kwargs):
        # expects: (B, D, L)
        if isinstance(x, tuple):
            x, aux = x
            x_dim = x.shape[-1]
            x = torch.cat([x, aux], dim=-1)
            res = self.ff(x.unsqueeze(-1)).squeeze(-1)
            res, res_aux = res[..., :x_dim], res[..., x_dim:]
            return (res, res_aux), state
        else:
            return self.ff(x.unsqueeze(-1)).squeeze(-1), state


class ResidualBlock(nn.Module):

    def __init__(
        self,
        d_model,
        layer,
        aux_layer=None,
        d_temb=0,
        dropout=0.0,
    ):
        """
        Residual S4 block.
        Args:
            d_model: dimension of the model
            bidirectional: use bidirectional S4 layer
            glu: use gated linear unit in the S4 layer
            dropout: dropout rate
        """
        super().__init__()

        self.layer = layer
        self.d_temb = d_temb
        self.aux_layer = aux_layer
        if self.aux_layer is not None:

            self.comb = nn.GLU(dim=1)

        if d_temb > 0:
            self.temb_proj = nn.Linear(d_temb, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x, t=None):
        """
        Input x is shape (B, d_input, L)
        """
        if isinstance(x, tuple):
            has_aux = True
            x, aux = x
            z_aux = aux
        else:
            has_aux = False

        z = x

        # Prenorm
        z = self.norm(z.transpose(-1, -2)).transpose(-1, -2)

        # Apply layer: we ignore the state input and output for training
        if has_aux:
            z_aux = self.norm(z_aux.transpose(-1, -2)).transpose(-1, -2)
            if self.aux_layer is not None:
                z_, _ = self.layer(z)
                z_aux_, _ = self.aux_layer(z_aux)
                z = self.comb(torch.cat([z_,z_aux_],dim=1))
                z_aux = self.comb(torch.cat([z_aux_, z_],dim=1))
            else:
                (z, z_aux), _ = self.layer((z, z_aux))
            z_aux = self.dropout(z_aux)
            aux = z_aux + aux
        else:
            z, _ = self.layer(z)

        if t is not None and self.d_temb > 0:
            z = z + self.temb_proj(t).transpose(-1,-2)[None]
        # Dropout on the output of the layer
        z = self.dropout(z)

        # Residual connection
        x = z + x
        if has_aux:
            return (x, aux), t, None
        else:
            return x,t, None

    def default_state(self, *args, **kwargs):
        if self.aux_layer is not None:
            return [self.layer.default_state(*args, **kwargs),
                    self.aux_layer.default_state(*args, **kwargs)]
        else:
            return self.layer.default_state(*args, **kwargs)

    def step(self, x, state, t=None, **kwargs):
        if isinstance(x, tuple):
            has_aux = True
            x, aux = x
            z_aux = aux
        else:
            has_aux = False
        z = x

        # Prenorm
        z = self.norm(z)

        # Apply layer
        if has_aux:
            z_aux = self.norm(z_aux)
            if self.aux_layer is not None:
                z_,  state[0] = self.layer.step(z, state[0], **kwargs)
                z_aux_,  state[1] = self.aux_layer.step(z_aux,  state[1], **kwargs)
                z = self.comb(torch.cat([z_,z_aux_],dim=1))
                z_aux = self.comb(torch.cat([z_aux_, z_],dim=1))
            else:
                (z, z_aux), state = self.layer.step((z, z_aux), state, **kwargs)

            aux = z_aux + aux
        else:
            z, state = self.layer.step(z, state,  **kwargs)


        if t is not None and self.d_temb > 0:
            z = z + self.temb_proj(t).transpose(-1,-2).squeeze(-1)[None]
        # Residual connection
        x = z + x

        if has_aux:
            return (x, aux), state
        else:
            return x, state


class SequentialUnet(nn.Module):

    def __init__(
        self,
        d_model=64,
        d_state=64,
        d_temb=0,
        n_layers=8,
        pool=[4, 4],
        expand=2,
        ff=2,
        bidirectional=False,
        # glu=True,
        unet=False,
        s4_type='s4',
        dropout=0.0,
        use_aux=False,
        mix_aux=True,
    ):
        """
        SaShiMi model backbone.
        Args:
            d_model: dimension of the model. We generally use 64 for all our experiments.
            n_layers: number of (Residual (S4) --> Residual (FF)) blocks at each pooling level.
                We use 8 layers for our experiments, although we found that increasing layers even further generally
                improves performance at the expense of training / inference speed.
            pool: pooling factor at each level. Pooling shrinks the sequence length at lower levels.
                We experimented with a pooling factor of 4 with 1 to 4 tiers of pooling and found 2 tiers to be best.
                It's possible that a different combination of pooling factors and number of tiers may perform better.
            expand: expansion factor when pooling. Features are expanded (i.e. the model becomes wider) at lower levels of the architecture.
                We generally found 2 to perform best (among 2, 4).
            ff: expansion factor for the FF inverted bottleneck. We generally found 2 to perform best (among 2, 4).
            bidirectional: use bidirectional S4 layers. Bidirectional layers are suitable for use with non-causal models
                such as diffusion models like DiffWave.
            glu: use gated linear unit in the S4 layers. Adds parameters and generally improves performance.
            unet: use a unet-like architecture, adding (Residual (S4) --> Residual (FF)) layers before downpooling.
                All else fixed, this slows down inference (and slightly slows training), but generally improves performance.
                We use this variant when dropping in SaShiMi into diffusion models, and this should generally be preferred
                for non-autoregressive models.
            dropout: dropout rate. Default to 0.0, since we haven't found settings where SaShiMi overfits.
        """
        super().__init__()
        self.d_model = H = d_model
        self.unet = unet
        self.n_layers = n_layers
        self.ff = ff
        self.s4_type = s4_type
        self.use_aux = use_aux
        self.mix_aux = mix_aux

        def s4_block(dim, tdim):
            if ( mix_aux and use_aux):
                if (s4_type == 's4d_joint'):
                    postact = 'glu'
                    aux_layer= None
                else:
                    postact = None
                    aux_layer = s4_registry[s4_type](
                        d_model=dim,
                        d_state=d_state,
                        bidirectional=bidirectional,
                        postact=postact,
                        dropout=dropout,
                        transposed=True,
                    )
            else:
                postact = 'glu'
                aux_layer= None
            layer = s4_registry[s4_type](
                d_model=dim,
                d_state=d_state,
                bidirectional=bidirectional,
                postact=postact,
                dropout=dropout,
                transposed=True,
            )

            return ResidualBlock(
                d_model=dim,
                layer=layer,
                aux_layer=aux_layer,
                d_temb=tdim,
                dropout=dropout,
            )

        def ff_block(dim, tdim):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=dropout,
                use_aux = use_aux
            )

            return ResidualBlock(
                d_model=dim,
                layer=layer,
                aux_layer=None,
                d_temb=tdim,
                dropout=dropout,
            )

        # Down blocks
        d_layers = []
        for p in pool:
            if unet:
                # Add blocks in the down layers
                for _ in range(n_layers):
                    d_layers.append(s4_block(H, d_temb))
                    if ff > 0: d_layers.append(ff_block(H, d_temb))

            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(H, expand, p, use_aux=use_aux))
            H *= expand
            d_temb *= p


        # Center block
        c_layers = []
        for _ in range(n_layers):
            c_layers.append(s4_block(H, d_temb))
            if ff > 0: c_layers.append(ff_block(H, d_temb))

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            block = []
            H //= expand
            d_temb //= p
            block.append(UpPool(H * expand, expand, p, use_aux=use_aux))

            for _ in range(n_layers):
                block.append(s4_block(H, d_temb))
                if ff > 0: block.append(ff_block(H, d_temb))

            u_layers.append(nn.ModuleList(block))

        self.d_layers = nn.ModuleList(d_layers)
        self.c_layers = nn.ModuleList(c_layers)
        self.u_layers = nn.ModuleList(u_layers)
        self.norm = nn.LayerNorm(H)

        assert H == d_model

    def forward(self, x, state=None, t=None):
        """
        input: (batch, length, d_input)
        output: (batch, length, d_output)
        """
        has_aux = False
        if isinstance(x, tuple):
            assert self.use_aux
            assert self.s4_type == 's4d_joint' or self.mix_aux
            has_aux = True
            x, aux = x
            aux = aux.transpose(1, 2)

        x = x.transpose(1, 2)

        # Down blocks
        outputs = []
        outputs.append(x)
        if has_aux:
            aux_outputs = []
            aux_outputs.append(aux)
        for layer in self.d_layers:
            if has_aux:
                (x, aux), t,_ = layer((x, aux), t=t)
                aux_outputs.append(aux)
            else:
                x, t, _ = layer(x, t=t)
            outputs.append(x)

        # Center block
        for layer in self.c_layers:
            if has_aux:
                (x, aux),t, _ = layer((x, aux), t=t)
            else:
                x,t, _ = layer(x, t=t)
        x = x + outputs.pop() # add a skip connection to the last output of the down block
        if has_aux:
            aux = aux + aux_outputs.pop()

        # Up blocks
        for block in self.u_layers:
            if self.unet:
                for layer in block:
                    if has_aux:
                        (x, aux), t, _ = layer((x, aux), t=t)
                        aux = aux + aux_outputs.pop()
                    else:
                        x,t, _ = layer(x, t=t)
                    x = x + outputs.pop() # skip connection
            else:
                for layer in block:
                    if has_aux:
                        (x, aux), t, _ = layer((x, aux), t=t)
                    else:
                        x, t, _ = layer(x, t=t)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                        if has_aux:
                            aux = aux + aux_outputs.pop()
                            aux_outputs.append(aux)

                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block
                if has_aux:
                    aux = aux + aux_outputs.pop()
        # feature projection
        x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        if has_aux:
            aux = aux.transpose(1, 2)
            aux = self.norm(aux)
            return (x, aux), None
        else:
            return x, None # required to return a state

    def default_state(self, *args, **kwargs):
        layers = list(self.d_layers) + list(self.c_layers) + [layer for block in self.u_layers for layer in block]
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, t=None, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        has_aux = False
        if isinstance(x, tuple):
            assert self.use_aux
            assert self.s4_type == 's4d_joint' or self.mix_aux
            has_aux = True
            x, aux = x
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SaShiMi
        next_state = []
        if has_aux:
            aux_outputs = []
        for layer in self.d_layers:
            outputs.append(x)

            if has_aux:
                aux_outputs.append(aux)
                x, _next_state = layer.step((x, aux), state=state.pop(), t=t, **kwargs)
            else:
                x, _next_state = layer.step(x, state=state.pop(), t=t, **kwargs)
            next_state.append(_next_state)
            if x is None: break
            if has_aux:
                x, aux = x

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped + len(self.c_layers)):
                next_state.append(state.pop())
            if self.unet:
                for i in range(skipped):
                    next_state.append(state.pop())
                # breakpoint()
                u_layers = list(self.u_layers)[skipped//((1+(self.ff>0))*self.n_layers + 1):]
            else:
                for i in range(skipped):
                    for _ in range(len(self.u_layers[i])):
                        next_state.append(state.pop())
                u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            if has_aux:
                aux_outputs.append(aux)

            for layer in self.c_layers:
                if has_aux:
                    (x, aux), _next_state = layer.step((x, aux), state=state.pop(), t=t, **kwargs)
                else:
                    x, _next_state = layer.step(x, state=state.pop(), t=t, **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            if has_aux:
                aux = aux + aux_outputs.pop()
            u_layers = self.u_layers

        for block in u_layers:
            if self.unet:
                for layer in block:
                    if has_aux:
                        (x, aux), _next_state = layer.step((x, aux), state=state.pop(), t=t, **kwargs)
                    else:
                        x, _next_state = layer.step(x, state=state.pop(), t=t, **kwargs)

                    next_state.append(_next_state)
                    x = x + outputs.pop()
                    if has_aux:
                        aux = aux + aux_outputs.pop()
            else:
                for layer in block:
                    if has_aux:
                        (x, aux), _next_state = layer.step((x, aux), state=state.pop(),  t=t,**kwargs)
                    else:
                        x, _next_state = layer.step(x, state=state.pop(), t=t, **kwargs)
                    next_state.append(_next_state)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                        if has_aux:
                            aux = aux + aux_outputs.pop()
                            aux_outputs.append(aux)
                x = x + outputs.pop()

                if has_aux:
                    aux = aux + aux_outputs.pop()
        # feature projection
        x = self.norm(x)

        if has_aux:
            aux = self.norm(aux)
            return (x, aux), next_state
        else:
            return x, next_state

