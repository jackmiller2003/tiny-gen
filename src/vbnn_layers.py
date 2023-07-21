from random import sample
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from abc import ABC


class StableSqrt(torch.autograd.Function):
    """
    Workaround to avoid the derivative of sqrt(0)
    This method returns sqrt(x) in its forward pass and in the backward pass
    it returns the gradient of sqrt(x) for all cases except for sqrt(0) where
    it returns the gradient 0
    """

    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors[0]
        grad = grad_output / (2.0 * result)
        grad[result == 0] = 0

        return grad


class VariationalLayer(nn.Module, ABC):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def compute_variational_kl_term(self):
        q_post, q_mean, q_var = self.get_post_params()
        p_prior = self.prior_dist
        return kl_divergence(q_post, p_prior).sum()


class VariationalConv2dLayer(VariationalLayer):
    def forward(self, inputs, local=True):
        if local:
            return self._forward_local(inputs)
        else:
            return self._forward_nonlocal(inputs)

    def _forward_local(self, inputs):
        q_post, q_mean, q_var = self.get_post_params()
        stride, padding = self.stride, self.padding
        mz = F.conv2d(inputs, q_mean, stride=stride, padding=padding)
        vz = F.conv2d(inputs.pow(2), q_var, stride=stride, padding=padding)
        eps = torch.empty(vz.size(), device=vz.device).normal_(0.0, 1.0)
        z_samples = eps * StableSqrt.apply(vz) + mz
        return z_samples

    def _forward_nonlocal(self, inputs):
        params = self.get_weight_samples()
        z_samples = F.conv2d(inputs, params, self.stride, self.padding)
        return z_samples

    def create_sample_module(self):
        params = self.get_weight_samples().squeeze(0)
        module = nn.Conv2d(
            self.size[1],
            self.size[0],
            self.size[2],
            self.stride,
            self.padding,
            bias=False,
        )
        module.weight.data = params.float()
        return module

    def extra_repr(self) -> str:
        s = "in_channels={}, out_channels={}".format(self.size[1], self.size[0])
        s += ", kernel_size=({}, {})".format(self.size[2], self.size[3])
        s += ", stride={}, padding={}".format(self.stride, self.padding)
        return s


class VariationalLinearLayer(VariationalLayer):
    def forward(self, inputs, local=True):
        if self.bias:
            onevec = torch.ones(inputs.shape[0], 1)
            if self.use_cuda:
                onevec = onevec.cuda()
            inputs = torch.cat([inputs, onevec], 1)
        if local:
            return self._forward_local(inputs)
        else:
            return self._forward_nonlocal(inputs)

    def _forward_local(self, inputs):
        q_post, q_mean, q_var = self.get_post_params()
        mz = torch.einsum("ni,io->no", inputs, q_mean)
        vz = torch.einsum("ni,io->no", inputs**2, q_var)
        eps = torch.empty(vz.size(), device=vz.device).normal_(0.0, 1.0)
        z_samples = eps * StableSqrt.apply(vz) + mz
        return z_samples

    def _forward_nonlocal(self, inputs):
        params = self.get_weight_samples()
        z_samples = torch.einsum("ni,io->no", inputs, params.squeeze(0))
        return z_samples

    def extra_repr(self) -> str:
        s = "in_features={}, out_features={}".format(self.size[0], self.size[1])
        return s


class GaussianLayer(nn.Module):
    def __init__(self, size, prior_mean=0.0, prior_std=1.0):
        super().__init__()
        self.size = S = torch.Size(size)
        self.use_cuda = use_cuda = torch.cuda.is_available()

        # prior
        prior_mean = torch.ones(S) * prior_mean
        prior_std = torch.ones(S) * prior_std
        if use_cuda:
            prior_mean = prior_mean.cuda()
            prior_std = prior_std.cuda()
        self.prior_dist = Independent(Normal(loc=prior_mean, scale=prior_std), 2)

        # variational parameters
        self.q_mean = nn.Parameter(torch.Tensor(S).normal_(0.0, 0.1))
        self.q_log_std = nn.Parameter(torch.log(torch.Tensor([0.01])) * torch.ones(S))

    def get_post_params(self):
        q_mean = self.q_mean
        q_std = torch.exp(self.q_log_std)
        q_dist = Independent(Normal(loc=q_mean, scale=q_std), 2)
        return q_dist, q_mean, q_std**2

    def weight_sample(self, no_samples=1):
        q_post, q_mean, q_var = self.get_post_params()
        params = q_post.sample(torch.Size([no_samples]))
        if self.use_cuda:
            params = params.cuda()
        return params


class GaussianLinear(VariationalLinearLayer, GaussianLayer):
    def __init__(self, size, prior_mean=0.0, prior_std=1.0, bias=True):
        VariationalLinearLayer.__init__(self)
        din = size[0]
        dout = size[1]
        if bias:
            din = din + 1
        sizeb = [din, dout]
        self.bias = bias
        GaussianLayer.__init__(self, sizeb, prior_mean, prior_std)


class GaussianConv2d(VariationalConv2dLayer, GaussianLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        prior_mean=0.0,
        prior_std=1.0
    ):
        VariationalConv2dLayer.__init__(self)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        size = [out_channels, in_channels, *kernel_size]
        GaussianLayer.__init__(self, size, prior_mean, prior_std)
        self.stride = stride
        self.padding = padding
