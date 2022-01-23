import torch
from torch import nn
import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.sample import Sampler
from torch import distributions as dist
from torch.autograd import grad
from model.flow import get_point_cnf
from utils.general import Embedder
import utils.plots as plt
import itertools, random
from torch import distributions as dist
import logging


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class LinearGrad(nn.Linear):
    def forward(self, input, input_grad, compute_grad=False, is_first=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None

        output_grad = self.weight[:, :3] if is_first else self.weight.matmul(input_grad)

        return output, output_grad


class TanHGrad(nn.Tanh):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = (1 - torch.tanh(input).pow(2)).unsqueeze(-1) * input_grad
        return output, output_grad


class SoftplusGrad(nn.Softplus):
    def forward(self, input, input_grad, compute_grad=False):
        output = super().forward(input)
        if not compute_grad:
            return output, None
        output_grad = torch.sigmoid(self.beta * input).unsqueeze(-1) * input_grad
        return output, output_grad


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, with_vae=True):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, c_dim)

        # torch.nn.init.constant_(self.fc.weight,0)
        # torch.nn.init.constant_(self.fc.bias, 0)

        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        if (with_vae):
            torch.nn.init.constant_(self.fc_mean.weight, 0)
            torch.nn.init.constant_(self.fc_mean.bias, 0)

            torch.nn.init.constant_(self.fc_std.weight, 0)
            torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        # latent = self.fc(self.actvn(net))

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean, c_std


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            latent_size,
            dims,
            dropout=None,
            dropout_prob=0.0,
            norm_layers=(),
            latent_in=(),
            weight_norm=False,
            activation=None,
            latent_dropout=False,
            xyz_dim=3,
            with_emb=False,
            is_v=False,
            geometric_init=True,
            beta=100
    ):
        super().__init__()

        bias = 1.0
        self.is_v = is_v
        self.latent_size = latent_size
        # dims = [d_in + latent_size] + dims + [d_out + feature_vector_size]
        if is_v:
            last_out_dim = latent_size * xyz_dim if latent_size > 0 else xyz_dim
        else:
            last_out_dim = 1
        dims = [latent_size + xyz_dim] + dims + [last_out_dim]
        self.d_in = latent_size + xyz_dim

        self.embed_fn = None
        multires = 0
        if (with_emb):
            multires = 6
            embed_fn, input_ch = Embedder.get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + latent_size
        self.latent_in = latent_in

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = LinearGrad(dims[l], out_dim)
            #lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    if (with_emb):
                        torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                        torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    else:
                        torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.latent_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=0, std=0.00001)
                    torch.nn.init.constant_(lin.bias, 0)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = SoftplusGrad(beta=100)
        # nn.ReLU()#
        #self.softplus = nn.Softplus(beta=beta)

    def forward(self, input, latent, compute_grad=False, cat_latent=True, grad_with_repsect_to_latent=False):
        '''
        :param input: [shape: (N x d_in)]
        :param compute_grad: True for computing the input gradient. default=False
        :return: x: [shape: (N x d_out)]
                 x_grad: input gradient if compute_grad=True [shape: (N x d_in x d_out)]
                         None if compute_grad=False
        '''

        # input.requires_grad_(True)
        # if self.embed_fn is not None:
        #     input_emb = self.embed_fn(input)
        # else:
        #     input_emb = input

        x = input
        input_con = latent.unsqueeze(1).repeat(1, input.shape[1], 1) if self.latent_size > 0 else input
        if self.latent_size > 0 and cat_latent:
            x = torch.cat([x, input_con], dim=-1) if len(x.shape) == 3 else torch.cat(
                [x, latent.repeat(input.shape[0], 1)], dim=-1)
        input_con = x
        to_cat = x
        x_grad = None

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.latent_in:
                x = torch.cat([x, to_cat], -1) / np.sqrt(2)
                if compute_grad:
                    skip_grad = torch.eye(self.d_in, device=x.device)[:, :3].unsqueeze(0).repeat(input.shape[0],input.shape[1], 1, 1)
                    x_grad = torch.cat([x_grad, skip_grad], 2) / np.sqrt(2)

            x, x_grad = lin(x, x_grad, compute_grad, l == 0)
            #x = lin(x)

            if l < self.num_layers - 2:
                x, x_grad = self.softplus(x, x_grad, compute_grad)
                #x = self.softplus(x)
        # if compute_grad:
        #     y = x
        #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        #     gradients = torch.autograd.grad(
        #         outputs=y,
        #         inputs=(input_con if grad_with_repsect_to_latent else input),
        #         grad_outputs=d_output,
        #         create_graph=True,
        #         retain_graph=True,
        #         only_inputs=True)[0]
        #     x_grad = gradients
        return x, x_grad, input_con


class DeformNetwork(nn.Module):
    def __init__(self, conf, latent_size, auto_decoder):
        super().__init__()
        self.with_vae = conf.get_bool('with_vae')
        self.with_sample = conf.get_bool('with_sample')
        self.latent_size = latent_size
        self.with_normals = conf.get_bool('encoder.with_normals')
        encoder_input_size = 6 if self.with_normals else 3

        self.encoder = SimplePointnet(hidden_dim=2 * latent_size, c_dim=latent_size, dim=encoder_input_size,
                                      with_vae=self.with_vae) if not auto_decoder and latent_size > 0 else None

        self.implicit_map = ImplicitNetwork(latent_size=latent_size, **conf.get_config('decoder_implicit'))

        self.v_latent_size = latent_size
        self.v_network = ImplicitNetwork(latent_size=self.v_latent_size, **conf.get_config('decoder_v'))
        self.predict_normals_on_surfce = conf.get_bool('predict_normals_on_surfce')
        self.sample_box = conf.get_list('sample_box')
        self.dist = dist.Normal(torch.tensor([]), torch.tensor([]))
        self.rand_sample_factor = conf.get_int('rand_sample_factor')
        self.dist_sample_factor = conf.get_int('dist_sample_factor')
        self.v_sample_factor = conf.get_int('v_sample_factor')
        self.v_start_uniform = conf.get_bool('v_start_uniform')
        self.dist_start_uniform = conf.get_bool('dist_start_uniform')
        self.v_projection_steps = conf.get_int('v_projection_steps')
        self.proj_with_con = conf.get_bool('proj_with_con')
        self.uniform_sample = conf.get_bool('uniform_sample')
        self.v_filter = conf.get_float('v_filter')
        self.t_samples = conf.get_int('t_samples')
        self.v_noise = conf.get_float('v_noise')
        self.con_dir = conf.get_bool('con_dir')
        self.dir_detach = conf.get_bool('dir_detach')
        self.concate_proj = conf.get_bool('concate_proj')
        self.t_include_bndry = conf.get_bool('t_include_bndry')
        self.t_beta_sampling = conf.get_bool('t_beta_sampling')
        self.is_nearest = conf.get_bool('is_nearest')
        self.is_mean_shape = conf.get_bool('is_mean_shape')

        logging.debug("""self.v_latent_size = {0},
                      self.predict_normals_on_surfce = {1}
                      self.sample_box = {2}
                      self.rand_sample_factor = {3}
                      self.dist_sample_factor = {4}
                      self.v_sample_factor = {5}
                      self.v_start_unfirom = {6}
                      self.v_projection_steps = {7}
                      self.uniform_sample = {8}
                      self.dist_start_unfirom = {9}
                      self.proj_with_con = {10}
                      self.v_filter = {11}
                      self.t_samples = {12}
                      self.v_noise = {13}
                      self.con_dir = {14}
                      self.concate_proj = {15}
                      self.t_include_bndry = {16}""".format(self.v_latent_size,
                                                            self.predict_normals_on_surfce,
                                                            self.sample_box,
                                                            self.rand_sample_factor,
                                                            self.dist_sample_factor,
                                                            self.v_sample_factor,
                                                            self.v_start_uniform,
                                                            self.v_projection_steps,
                                                            self.uniform_sample,
                                                            self.dist_start_uniform,
                                                            self.proj_with_con,
                                                            self.v_filter,
                                                            self.t_samples,
                                                            self.v_noise,
                                                            self.con_dir,
                                                            self.concate_proj,
                                                            self.t_include_bndry))

    def forward(self, manifold_points, manifold_normals, sample_nonmnfld, latent, latent_sigma_inputs,
                only_encoder_forward, only_decoder_forward, spider_head=None, is_additional_latent=False, z_func=None,
                epoch=-1):
        output = {}

        if self.encoder is not None and not only_decoder_forward:
            encoder_input = torch.cat([manifold_points, manifold_normals],
                                      axis=-1) if self.with_normals else manifold_points
            q_latent_mean, q_latent_std = self.encoder(encoder_input)

            q_z = dist.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent = q_z.rsample()
            latent_reg = (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
            output['latent_reg'] = latent_reg

            if only_encoder_forward:
                return latent, q_latent_mean, torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None, None

        if only_decoder_forward:
            return self.implicit_map(manifold_points, latent, False)[0]
        else:

            reconstructed_surface, non_mnfld_grad, _ = self.implicit_map(sample_nonmnfld, latent, True)
            rand_sample = self.get_rand_sample(manifold_points, latent.shape[0], manifold_points.shape[1]//2).detach()
            _,latent_eiko_grad,_ = self.implicit_map(rand_sample, latent, True)

            output['nonmnfld_grad'] = non_mnfld_grad
            output['recon_surface'] = reconstructed_surface
            output['latent_eiko_grad'] = (latent_eiko_grad**2).sum(-1)#torch.norm(latent_eiko_grad,p=2,dim=-1)# (latent_eiko_grad**2).sum(-1)

            return output

    def get_sample_in_bounding_box(self, pnts, batch_size, num_points):

        s_mean = pnts.detach().mean(dim=0, keepdim=True)
        s_cov = pnts - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
        if torch.det(vecs) < 0:
            vecs = torch.mm(utils.get_cuda_ifavailable(torch.tensor([[0, 1], [1, 0]])).float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1),
                           (pnts.detach() - s_mean).unsqueeze(-1))
        max_x = helper[:, 0, 0].max().item() + 0.2
        min_x = helper[:, 0, 0].min().item() - 0.2
        max_y = helper[:, 1, 0].max().item() + 0.2
        min_y = helper[:, 1, 0].min().item() - 0.2
        scale_tensor = torch.tensor([[max_x - min_x, 0], [0, max_y - min_y]]).to(pnts)
        sample = torch.rand([batch_size, pnts.shape[-1], num_points]).to(pnts)
        helper2 = torch.bmm(scale_tensor.unsqueeze(0).repeat(batch_size, 1, 1), sample) + torch.tensor(
            [min_x, min_y]).unsqueeze(0).unsqueeze(-1).to(pnts)

        sample = (torch.bmm(vecs.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1, 2), helper2) + s_mean.unsqueeze(
            -1)).detach()
        sample = sample.permute(0, 2, 1)
        return sample

    def get_rand_sample(self, pnts, batch_size, num_points):

        rand_pnts = torch.rand([batch_size, num_points, pnts.shape[-1]]).to(pnts)
        max_x = self.sample_box[0]
        min_x = self.sample_box[1]
        max_y = self.sample_box[2]
        min_y = self.sample_box[3]
        if len(self.sample_box) == 3 * 2:
            max_z = self.sample_box[4]  # 1.2
            min_z = self.sample_box[5]  # -1.2
            scale_tensor = torch.tensor([[max_x - min_x, 0, 0], [0, max_y - min_y, 0], [0, 0, max_z - min_z]]).to(
                rand_pnts)
            rand_pnts = torch.bmm(scale_tensor.unsqueeze(0).repeat(rand_pnts.shape[0], 1, 1),
                                  rand_pnts.transpose(2, 1)) + torch.tensor([min_x, min_y, min_z]).unsqueeze(
                0).unsqueeze(-1).to(rand_pnts)
        else:
            scale_tensor = torch.tensor([[max_x - min_x, 0], [0, max_y - min_y]]).to(rand_pnts)
            rand_pnts = torch.bmm(scale_tensor.unsqueeze(0).repeat(rand_pnts.shape[0], 1, 1),
                                  rand_pnts.transpose(2, 1)) + torch.tensor([min_x, min_y]).unsqueeze(0).unsqueeze(
                -1).to(rand_pnts)
        rand_pnts = rand_pnts.transpose(1, 2)
        return rand_pnts



