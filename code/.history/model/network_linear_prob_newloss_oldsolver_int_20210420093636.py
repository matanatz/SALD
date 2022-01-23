import torch
from torch import nn
import numpy as np
import utils.general as utils
import torch.nn as nn
import torch
#from pytorch3d.ops import knn_points
import time
import itertools, random
from torch import distributions as dist
import logging
from model.implicit_map import ImplicitNetwork


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out



class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        
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

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean,c_std


class DeformNetwork(nn.Module):
    def __init__(self, conf, latent_size, auto_decoder):
        super().__init__()
        self.conf = conf
        self.latent_size = latent_size
        self.with_normals = conf.get_bool('encoder.with_normals')
        encoder_input_size = 6 if self.with_normals else 3
        self.encoder = SimplePointnet(hidden_dim=2 * latent_size, c_dim=latent_size, dim=encoder_input_size, with_vae=self.with_vae) if not auto_decoder else None
        self.implicit_map = ImplicitNetwork(latent_size=latent_size,last_out_dim=1, **conf.get_config('decoder_implicit'))
        self.predict_normals_on_surfce = conf.get_bool('predict_normals_on_surfce')
        
        logging.debug("""self.latent_size = {0},
                      self.with_normals = {1},
                      self.predict_normals_on_surfce = {2}
                      """.format(self.latent_size,self.with_normals,self.predict_normals_on_surfce))

    def forward(self, manifold_points, manifold_normals, sample_nonmnfld, latent, latent_sigma_inputs, only_encoder_forward, only_decoder_forward,epoch=-1,external_sample=None,external_latent=None):
        output = {}

        if self.encoder is not None and not only_decoder_forward:
            encoder_input = torch.cat([manifold_points, manifold_normals],axis=-1) if self.with_normals else manifold_points
            q_latent_mean,q_latent_std = self.encoder(encoder_input)

            q_z = torch.distributions.Normal(q_latent_mean, torch.exp(q_latent_std))
            latent_draw = q_z.rsample()
            latent_reg = (q_latent_mean.abs().mean(dim=-1) + (q_latent_std + 1).abs().mean(dim=-1))
            output['latent_reg'] = latent_reg
            latent = q_latent_mean
            
            if self.dir_detach:
                dirs = dirs.detach()

            if only_encoder_forward:
                return latent,q_latent_mean,torch.exp(q_latent_std)
        else:
            if only_encoder_forward:
                return None,None
            

        if only_decoder_forward:
            return self.implicit_map(manifold_points,latent, False)[0]
        else:
            if self.with_emb:
                sample_nonmnfld = sample_nonmnfld.detach()
                sample_nonmnfld.requires_grad_(True)
            reconstructed_surface, non_mnfld_grad,_,_ = self.implicit_map(sample_nonmnfld, latent, True)

            if self.predict_normals_on_surfce:
                _, can_grad, _,_ = self.implicit_map(manifold_points, latent, True)
                output['grad_on_surface'] = can_grad
            else:
                output['nonmnfld_grad'] = non_mnfld_grad

            output['recon_surface'] = reconstructed_surface

            if not latent is None:
                output['norm_square_latent'] = (latent**2).mean(-1)

            # add latent samples to batch
            if (latent is not None) and ((latent.shape[0] > 1 or outside_latent is not None) and self.encoder is None):
                if self.shape_interpolation == 0:
                    if epoch == -1:
                        t_rand_sample = torch.rand(self.t_samples, len(latent.shape[0]),device=latent.device)
                    else:
                        if self.t_beta_sampling:
                            beta = np.linspace(0.01, 1, 10000)
                            if epoch >= 10000:
                                beta = 1
                            else:
                                beta = beta[epoch]
                        else:
                            beta = 1
                        t_rand_sample = torch.rand(self.t_samples, len(latent.shape[0]),device=latent.device) #torch.distributions.beta.Beta(beta, beta).sample([self.t_samples, latent.shape[0]])


                    t = t_rand_sample
                    sample_latent = t.unsqueeze(-1) * latent.unsqueeze(0)
                    dirs = (latent).unsqueeze(0).repeat(t.shape[0], 1, 1)
                    dirs = dirs.view(-1, dirs.shape[-1])
                    if self.dir_detach:
                        dirs = dirs.detach()
                    sample_latent = sample_latent.view(-1, sample_latent.shape[-1])
                elif self.shape_interpolation == 1:
                    if self.is_nearest:
                        dist = ((latent ** 2).sum(-1, keepdims=True) - 2 * torch.mm(latent, latent.T) + (latent ** 2).sum(
                            -1, keepdims=True).T).detach()
                        closest = dist.topk(k=9, dim=-1, largest=False)
                        idx = torch.randint(low=1,high=9,size=[latent.shape[0]])
                        second_idx = torch.gather(closest[1],dim=1,index=idx.to(latent).long().unsqueeze(-1)).squeeze()
                        first_idx = torch.gather(closest[1],dim=1,index=torch.zeros([latent.shape[0]]).to(latent).long().unsqueeze(-1)).squeeze()
                        number_of_pairs = latent.shape[0]
                    else:
                        if outside_latent is None:
                            g = itertools.combinations(range(latent.shape[0]),2)
                            alist = list(g)
                            random.shuffle(alist)
                            alist = alist[:latent.shape[0]]
                            fst = np.random.randint(0,2)
                            first_idx = [x[fst] for x in alist]
                            second_idx = [x[(fst + 1) %2] for x in alist]
                            number_of_pairs = len(alist)
                        else:
                            number_of_pairs = outside_latent.shape[0]
                            first_idx = [0]
                            second_idx = range(1,outside_latent.shape[0]+1)
                            latent = torch.cat([latent,outside_latent],dim=0)


                    if epoch == -1:
                        t_rand_sample = torch.rand(self.t_samples,number_of_pairs,device=latent.device)
                    else:
                        if self.t_beta_sampling:
                            beta = np.linspace(0.01,1,10000)
                            if epoch >= 10000:
                                beta = 1
                            else:
                                beta = beta[epoch]
                        else:
                            beta = 1
                        t_rand_sample = torch.rand(self.t_samples,number_of_pairs,device=latent.device)#torch.distributions.beta.Beta(beta , beta).sample([self.t_samples,number_of_pairs])

                    if self.t_include_bndry:
                        t = torch.cat([torch.zeros([1,number_of_pairs]),torch.ones([1,number_of_pairs]),t_rand_sample],dim=0)
                    else:
                        t = t_rand_sample

                    sample_latent = latent[first_idx].unsqueeze(0) + t.unsqueeze(-1) * (latent[second_idx] - latent[first_idx]).unsqueeze(0)
                    dirs = (latent[second_idx] - latent[first_idx]).unsqueeze(0).repeat(t.shape[0],1,1)
                    dirs = dirs.view(-1,dirs.shape[-1])
                    if self.dir_detach:
                        dirs = dirs.detach()
                    sample_latent = sample_latent.view(-1,sample_latent.shape[-1])
                elif self.shape_interpolation == 2:
                    if outside_latent is None:
                        g = itertools.combinations(range(latent.shape[0]),2)
                        alist = list(g)
                        random.shuffle(alist)
                        alist = alist[:latent.shape[0]]
                        fst = np.random.randint(0,2)
                        first_idx = [x[fst] for x in alist]
                        second_idx = [x[(fst + 1) %2] for x in alist]
                        number_of_pairs = len(alist)
                    else:
                        number_of_pairs = outside_latent.shape[0]
                        first_idx = [0]
                        second_idx = range(1,outside_latent.shape[0]+1)
                        latent = torch.cat([latent,outside_latent],dim=0)

                    if epoch == -1:
                        t_rand_sample = torch.rand(self.t_samples,number_of_pairs,device=latent.device)
                    else:
                        if self.t_beta_sampling:
                            beta = np.linspace(0.01,1,10000)
                            if epoch >= 10000:
                                beta = 1
                            else:
                                beta = beta[epoch]
                        else:
                            beta = 1
                        t_rand_sample = torch.rand(self.t_samples,number_of_pairs,device=latent.device)#torch.distributions.beta.Beta(beta , beta).sample([self.t_samples,number_of_pairs])

                    if self.t_include_bndry:
                        t = torch.cat([torch.zeros([1,number_of_pairs]),torch.ones([1,number_of_pairs]),t_rand_sample],dim=0)
                    else:
                        t = t_rand_sample

                    first_normalize = torch.nn.functional.normalize(latent[first_idx],p=2,dim=-1).unsqueeze(0)
                    second_normalize = torch.nn.functional.normalize(latent[second_idx],p=2,dim=-1).unsqueeze(0)
                    Omega = torch.acos((first_normalize * second_normalize).sum(-1,keepdim=True))

                    norm_int = torch.norm(latent[first_idx].unsqueeze(0),p=2,dim=-1,keepdim=True) + t.unsqueeze(-1) * (torch.norm(latent[second_idx],p=2,dim=-1,keepdim=True) - torch.norm(latent[first_idx],p=2,dim=-1,keepdim=True)).unsqueeze(0)

                    P_t = (torch.sin((1 - t.unsqueeze(-1)) * Omega) / torch.sin(Omega)) * first_normalize + (torch.sin((t.unsqueeze(-1)) * Omega) / torch.sin(Omega)) * second_normalize
                    P_dot_t = -(torch.cos((1 - t.unsqueeze(-1)) * Omega) * Omega / torch.sin(Omega)) * first_normalize + Omega * (torch.cos((t.unsqueeze(-1)) * Omega) / torch.sin(Omega)) * second_normalize
                    sample_latent = norm_int * P_t
                    
                    #sample_latent = latent[first_idx].unsqueeze(0) + t.unsqueeze(-1) * (latent[second_idx] - latent[first_idx]).unsqueeze(0)
                    dirs = (torch.norm(latent[second_idx],p=2,dim=1,keepdim=True) - torch.norm(latent[first_idx],p=2,dim=-1,keepdim=True)).unsqueeze(0).repeat(t.shape[0],1,1) * P_t + norm_int * P_dot_t
                    #dirs = (latent[second_idx] - latent[first_idx]).unsqueeze(0).repeat(t.shape[0],1,1)
                    dirs = dirs.view(-1,dirs.shape[-1])
                    if self.dir_detach:
                        dirs = dirs.detach()
                    sample_latent = sample_latent.view(-1,sample_latent.shape[-1])



            else:
                sample_latent = latent
            
            
            if self.adaptive_with_sample:
                with_sample = epoch > self.adaptive_epoch or epoch == -1
            else:
                with_sample = self.with_sample
            if (with_sample):
                sample_latent_size = sample_latent.shape[0]    
                if outside_sample is None:
                    curr_projection = self.get_rand_sample(manifold_points, sample_latent_size , manifold_points.shape[1]//self.v_sample_factor).detach()

                    if self.with_emb:
                        curr_projection.requires_grad_(True)
                    _, eiko_grad, _,_ = self.implicit_map(curr_projection, sample_latent, True)
                    output['latent_eiko_grad'] = (eiko_grad ** 2).sum(-1)

                    output['debug_v_projection_start'] = curr_projection
                    proj_latent = sample_latent.detach()
                    output['debug_latent'] = proj_latent

                    #start = time.time()

                    # Find bounding box
                    for i in range(3):
                        network_eval, grad,_,_ = self.implicit_map(curr_projection if self.with_emb else curr_projection.detach(), proj_latent, True)
                        network_eval = network_eval.detach()
                        grad = grad.detach()
                        sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True).detach()
                        curr_projection = curr_projection - (network_eval.abs() > 1.0e-4).repeat(1,1,3)*(network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)).squeeze(2))
                        curr_projection[:,:,0] = curr_projection[:,:,0].clamp(self.sample_box[1],self.sample_box[0])
                        curr_projection[:,:,1] = curr_projection[:,:,1].clamp(self.sample_box[3],self.sample_box[2])
                        curr_projection[:,:,2] = curr_projection[:,:,2].clamp(self.sample_box[5],self.sample_box[4])
                    # plot_surface(True,curr_projection[0].to(proj_latent),
                    #                 self,
                    #                 proj_latent[0],
                    #                 '.',
                    #                 0,0,
                    #                 'a',
                    #                 256,0,
                    #                 True,True,True,False,True,True)
                    
                    rand_pnts = torch.rand_like(curr_projection)
                    center = curr_projection.mean(1,keepdims=True)
                    curr_projection  = curr_projection - center
                    
                    max_bnd = curr_projection.max(dim=1).values + 0.05
                    min_bnd = curr_projection.min(dim=1).values - 0.05
                    max_min = max_bnd - min_bnd
                    
                    rand_pnts = torch.bmm(torch.diag_embed(max_min),rand_pnts.transpose(1,2)) + min_bnd.unsqueeze(-1)
                    curr_projection = rand_pnts.transpose(1,2).detach() + center
                    output['debug_v_projection_start'] = curr_projection
                    # logging.debug('after bndning_box {0}'.format(time.time() - start))
                    # start = time.time()

                    if self.adapted_lr:
                        lr = [0.1,0.25,1.0,1.0,1.0]
                    else:
                        lr = [1.0,1.0,1.0,1.0,1.0]
                    if self.proj_with_con:
                        for i in range(4):
                            network_eval, grad,_,_ = self.implicit_map(curr_projection.detach(), proj_latent, True)
                            network_eval = network_eval.detach()
                            grad = grad.detach()
                            sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True).detach()
                            curr_projection = curr_projection - (network_eval.abs() > self.v_filter).repeat(1,1,3)*(network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)))

                        curr_projection = curr_projection.detach()
                        curr_projection.requires_grad_(True)
                        proj_latent.requires_grad_(True)
                        ti = t.detach().to(latent)
                        ti.requires_grad_(True)
                        optim = torch.optim.SGD(params=[curr_projection,ti],lr=1.0e-1)
                        for i in range(5):
                            curr_proj_latent = latent[first_idx].unsqueeze(0).detach() + ti * (latent[second_idx] - latent[first_idx]).unsqueeze(0).detach()
                            curr_proj_latent = curr_proj_latent.view(-1,curr_proj_latent.shape[-1])
                            network_eval, _,_ = self.implicit_map(curr_projection, curr_proj_latent, False)
                            v_output = self.v_network(curr_projection, curr_proj_latent, False)[0]
                            v_output = v_output.reshape(v_output.shape[0], v_output.shape[1], -1,max(self.v_latent_size, 1))
                            v = torch.einsum('bpij,bj->bpi',v_output,dirs)
                            loss = (network_eval**2).sum() - 0.01*(v**2).sum()
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        print (network_eval.abs().mean())
                        proj_latent = latent[first_idx].unsqueeze(0) + ti.detach() * (
                                    latent[second_idx] - latent[first_idx]).unsqueeze(0)
                        proj_latent = proj_latent.view(-1, proj_latent.shape[-1])
                    else:

                        for i in range(self.v_projection_steps):
                            network_eval, grad,_,_ = self.implicit_map(curr_projection if self.with_emb else curr_projection.detach(), proj_latent, True)
                            network_eval = network_eval.detach()
                            grad = grad.detach()
                            sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True).detach()
                            lr_i = lr[i] if i < len(lr) else 1.0
                            curr_projection = curr_projection - lr_i*(network_eval.abs() > self.v_filter)*(network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)).squeeze(2))
                            curr_projection[:,:,0] = curr_projection[:,:,0].clamp(self.sample_box[1],self.sample_box[0])
                            curr_projection[:,:,1] = curr_projection[:,:,1].clamp(self.sample_box[3],self.sample_box[2])
                            curr_projection[:,:,2] = curr_projection[:,:,2].clamp(self.sample_box[5],self.sample_box[4])
                else:
                    dirs = outside_dir
                    proj_latent = latent.detach()
                    curr_projection = outside_sample
                network_eval, proj_grad,input_con,_ = self.implicit_map(curr_projection if self.with_emb else curr_projection.detach(), proj_latent, False,grad_with_repsect_to_latent=False)
                output['v_filter'] = (network_eval.abs() < self.v_filter).detach()

                if self.concate_proj:
                    curr_projection = torch.cat([curr_projection, (curr_projection + self.v_noise*torch.randn_like(curr_projection)).detach()],dim=1)
                    output['v_filter'] = torch.cat([output['v_filter'],output['v_filter']],dim=1)
                else:
                    curr_projection = (curr_projection + self.v_noise * torch.randn_like(curr_projection)).detach()

                output['debug_v_projection_end'] = curr_projection
                output['debug_v_projection_end_eval'] = network_eval
                #logging.debug('after projection {0}'.format(time.time() - start))

                curr_projection.requires_grad_(True)
                # if self.detach_f:
                #     sample_latent = sample_latent.detach()
                # v_output = self.v_network(curr_projection, sample_latent, False)[0]
                # v_output = v_output.reshape(v_output.shape[0], v_output.shape[1], -1, max(self.v_latent_size,1))
                # output['v_output'] = v_output

                #start = time.time()
                eval_proj, proj_grad,input_con,_ = self.implicit_map(curr_projection,
                                                                   sample_latent,
                                                                   False,
                                                                   grad_with_repsect_to_latent=True)
                
                prob = self.prob_net(torch.cat([curr_projection,sample_latent.unsqueeze(1).repeat(1,curr_projection.shape[1],1)],dim=-1))

                # #prob = torch.cat([curr_projection[...,2:3],-100*curr_projection[...,2:3]],dim=-1)
                # #prob = torch.cat([curr_projection[...,2:3],-10*curr_projection[...,2:3]],dim=-1)

                #prob = torch.nn.Softmax(dim=-1)(prob)
                prob =  torch.nn.Softmax(dim=-1)(prob)
                #prob = torch.nn.functional.one_hot(manifold_normals.to(torch.int64).squeeze(-1) - 1)
                
                output['clusters'] = prob
                dfdz = proj_grad[:,:,-self.latent_size:]
                output['dfdz'] = (dfdz**2).sum(-1)
                proj_grad = proj_grad[:,:,:manifold_points.shape[-1]]

                grad_filter = (proj_grad**2).sum(-1,keepdim=True) > 1.0e-1
                output['grad_filter'] = grad_filter

                # v_hat = - torch.einsum('bpk,bpj->bpkj',
                #                        proj_grad,
                #                        dfdz - eval_proj * torch.einsum('bpk,bpkj->bpj',proj_grad,v_output))/(proj_grad**2).sum(-1,keepdim=True).unsqueeze(-1)
                #
                # s = v_hat + (1)*v_output - (torch.einsum('bpkj,bpkj->bpj',
                #                                          v_output,
                #                                          proj_grad.unsqueeze(-1).repeat(1,1,1,v_output.shape[-1])) / (proj_grad**2).sum(-1,keepdims=True)).unsqueeze(2) * proj_grad.unsqueeze(-1).repeat(1,1,1,v_output.shape[-1])
                proj_grad_norm_square = (proj_grad ** 2).sum(-1, keepdims=True).clamp_min(1.0e-4)

                if dirs is None:
                    u = torch.randn(size=[dfdz.shape[0],dfdz.shape[1]]).to(proj_grad)
                    #u = torch.nn.functional.normalize(u,p=2,dim=-1)
                else:
                    u = dirs
                u = torch.nn.functional.normalize(u, p=2, dim=-1)

                W = torch.einsum('bpk,bpj->bpkj', proj_grad * (1. / proj_grad_norm_square),-dfdz)
                Wdz = torch.einsum('bpxz,bz->bpx',W,u)
                Wdzdx = torch.cat([torch.autograd.grad(Wdz[:,:,i],
                                                       curr_projection,
                                                       torch.ones_like(Wdz[:,:,i]),
                                                       retain_graph=True,
                                                       create_graph=True)[0].unsqueeze(2) for i in range(Wdz.shape[2])],dim=2)
                Wdzdx_ij_ji = (Wdzdx.transpose(2,3) + Wdzdx).reshape(Wdzdx.shape[0],Wdzdx.shape[1],-1).detach()

                # P = torch.eye(proj_grad.shape[-1],proj_grad.shape[-1]).to(proj_grad).repeat(proj_grad.shape[0],proj_grad.shape[1],1,1) +\
                #     (eval_proj - 1).unsqueeze(-1) * torch.einsum('bpk,bpi->bpki',proj_grad,proj_grad)/proj_grad_norm_square.unsqueeze(-1)
                P = torch.eye(proj_grad.shape[-1], proj_grad.shape[-1],device=proj_grad.device).repeat(proj_grad.shape[0],
                                                                                             proj_grad.shape[1], 1, 1) - proj_grad.unsqueeze(-1) * proj_grad.unsqueeze(-2) / proj_grad_norm_square.unsqueeze(-1)
                res = []
                C = []
                ei = torch.eye(3,device=P.device)
                for i in range(P.shape[2]):
                    #res.append([])
                    C.append([])
                    for l in range(P.shape[3]):
                        # res[-1].append(torch.autograd.grad(P[:,:,i,l],
                        #                     curr_projection,
                        #                     torch.ones_like(P[:,:,i,l]),
                        #                     retain_graph=True,
                        #                     create_graph=False)[0].unsqueeze(-1).detach())
                        C[-1].append(torch.einsum('bpkl,ld->bpkd',
                                                  P.transpose(2,3),
                                                  torch.mm(ei[:,i:(i+1)],ei[l:(l+1),:])).unsqueeze(2).detach())

                    #res[-1] = torch.cat(res[-1],dim=-1).unsqueeze(-2)
                    C[-1] = torch.cat(C[-1],dim=2).unsqueeze(2)

                #Pdx_old = torch.cat(res, dim=3)
                H =  torch.cat([torch.autograd.grad(proj_grad[...,i],curr_projection,torch.ones_like(proj_grad[...,i]),retain_graph=True,create_graph=True)[0].unsqueeze(2) for i in range(proj_grad.shape[-1])],dim=2)
                normal_d = torch.einsum('bpij,bpjk->bpik',P,H) # 1./proj_grad_norm.unsqueeze(-1)

                Pdx = -(1./proj_grad_norm_square.unsqueeze(-1).unsqueeze(-1)) * torch.einsum('bpjil,bpjk->bpkil',torch.eye(3,device=proj_grad.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0) * (proj_grad).unsqueeze(2).unsqueeze(2) + torch.eye(3,device=proj_grad.device).unsqueeze(1).unsqueeze(0).unsqueeze(0) * (proj_grad).unsqueeze(2).unsqueeze(-1),
                                                                                        normal_d)

                B_ij = torch.einsum('bpijl,bpk->bpijlk', Pdx.permute(0,1,3,2,4), curr_projection)
                C_ij = torch.cat(C, dim=2)
                B_C_ij_ji = B_ij.transpose(2, 3) + B_ij + C_ij + C_ij.transpose(2, 3)
                D_ij_ji = Pdx.permute(0,1,3,2,4) + Pdx
                X = torch.cat([B_C_ij_ji.flatten(-2), D_ij_ji],dim=-1).permute(0,1,4,2,3).flatten(-2).transpose(2,3)

                
                sol_A = []
                sol_b = []
                #residual = []
                # logging.debug('after lsq diff calc {0}'.format(time.time() - start))
                # start = time.time()
                for pi in range(prob.shape[-1]):
                    p_k = torch.sqrt(prob[...,pi]).detach()
                    
                    X_all = (p_k.unsqueeze(-1).unsqueeze(-1)*X[:,:,[0,1,2,4,5,8],:]).view(X.shape[0],-1, 12)
                    W_all = (p_k.unsqueeze(-1) * Wdzdx_ij_ji[:,:,[0,1,2,4,5,8]]).view(Wdzdx_ij_ji.shape[0],-1,1)

                    try:
                        x_inv = torch.pinverse(X_all.detach().cpu()).detach().to(X_all)
                    except:
                        x_inv = torch.pinverse((X_all).detach().cpu(),rcond=1e-4).detach().to(X_all)
                        logging.debug("exception!!!")

                    sol = torch.einsum('bpj,bjk->bpk',
                                    x_inv,
                                    -W_all)                                   
                    #residual.append(((torch.einsum('bpij,bj->bpi',X,sol.squeeze(-1)) + Wdzdx_ij_ji)**2).sum(-1,True))
                    sol[sol.isnan()] = 0
                    sol_A.append( sol[:,:9].view(sol.shape[0],1,3,3).repeat(1,curr_projection.shape[1],1,1).detach().unsqueeze(-1))
                    sol_b.append( sol[:,9:].view(sol.shape[0],1, 3).repeat(1,curr_projection.shape[1], 1).detach().unsqueeze(-1))

                sol_A = torch.cat(sol_A,dim=-1)
                sol_b = torch.cat(sol_b,dim=-1)

            
                # prob_t = prob.transpose(1,2).contiguous().detach()

                # xp = (X[:,:,[0,1,2,4,5,8],:].unsqueeze(1) * prob_t.unsqueeze(-1).unsqueeze(-1)).view(prob_t.shape[0],prob_t.shape[1],-1,12).detach()
                # A = torch.einsum('brpi,brpj->brij',X[:,:,[0,1,2,4,5,8],:].unsqueeze(1).repeat(1,prob_t.shape[1],1,1,1).view(prob_t.shape[0],prob_t.shape[1],-1,12),xp).detach()
                # W_all = (Wdzdx_ij_ji[:,:,[0,1,2,4,5,8]].unsqueeze(1) * prob_t.unsqueeze(-1) ).view(prob_t.shape[0],prob_t.shape[1],-1,1)
                # B = torch.einsum('brpi,brpk->brik',X[:,:,[0,1,2,4,5,8],:].unsqueeze(1).repeat(1,prob_t.shape[1],1,1,1).view(prob_t.shape[0],prob_t.shape[1],-1,12),-W_all).detach()
                # sol_new = torch.einsum('bpij,bpjk->bpik',torch.inverse(A + 1.e-5*torch.eye(A.shape[-1],device=A.device).unsqueeze(0).unsqueeze(0)),B).detach()
                # sol_A = sol_new[:,:,:9].view(sol_new.shape[0],sol_new.shape[1],3,3).permute(0,2,3,1).unsqueeze(1).repeat(1,curr_projection.shape[1],1,1,1).detach()#torch.cat(sol_A,dim=-1)
                # sol_b = sol_new[:,:,9:].permute(0,3,2,1).repeat(1,curr_projection.shape[1],1,1).detach()# torch.cat(sol_b,dim=-1)



                #output['residual'] = torch.cat(residual,dim=-1)
                s2 = (torch.einsum('bpijr,bpj->bpir',sol_A,curr_projection) + sol_b)
                s = torch.einsum('bpij,bpjr->bpir',P,s2)
                
                #logging.debug('after u calc {0}'.format(time.time() - start))
                #start = time.time()
                # res = []
                # res_2 = []
                # for i in range(s.shape[-1]):
                #     res.append([])
                #     res_2.append([])
                #     for l in range(s.shape[-2]):
                #         res[-1].append(torch.autograd.grad(s[:,:,l,i],
                #                                             curr_projection,
                #                                             torch.ones_like(s[:,:,l,i]),
                #                                             retain_graph=True,
                #                                             create_graph=True)[0].unsqueeze(-1))
                #         res_2[-1].append(torch.autograd.grad(s2[:,:,l,i],
                #                                             curr_projection,
                #                                             torch.ones_like(s2[:,:,l,i]),
                #                                             retain_graph=True,
                #                                             create_graph=True)[0].unsqueeze(-1))
                #     res[-1] = torch.cat(res[-1],dim=-1).unsqueeze(-2)
                #     res_2[-1] = torch.cat(res_2[-1],dim=-1).unsqueeze(-2)
                # res = torch.cat(res,dim=3).transpose(2,4)#.permute(0,3,1,2,4)
                # res_2 = torch.cat(res_2,dim=3)#.permute(0,3,1,2,4)
                # ds = torch.cat([torch.autograd.grad(s[:,:,i],
                #                          curr_projection,
                #                          prob,
                #                          retain_graph=True,
                #                          create_graph=True)[0].unsqueeze(2) for i in range(s.shape[2])],dim=2)
                ds = (torch.einsum('bpjik,bpkr->bpjir' ,Pdx,s2) + torch.einsum('bpij,bpjkr->bpkir',P,sol_A)).permute(0,1,3,4,2)
                #logging.debug('after ds calc {0}'.format(time.time() - start))
                # u_output = (prob.permute(2,0,1).unsqueeze(-1) * (torch.einsum('rbpij,bpj->rbpi',torch.cat(sol_A,dim=0),curr_projection) + torch.cat(sol_b,dim=0))).sum(0)

                # s = torch.einsum('bpij,bpj->bpi',P,u_output)

                # ds = torch.cat([torch.autograd.grad(s[:,:,i],
                #                          curr_projection,
                #                          torch.ones_like(s[:,:,i]),
                #                          retain_graph=True,
                #                          create_graph=True)[0].unsqueeze(2) for i in range(s.shape[2])],dim=2)
                #dv = Wdzdx + ds
                dv = Wdzdx.unsqueeze(3) + ds

                # Killing vector field term
                if dv.isnan().any():
                    print ('any')
                #output['dv'] = dv + dv.permute(0,1,3,2)
                output['dv_prob'] = (grad_filter.detach().float() * output['v_filter'].detach().float()).squeeze() *  (prob * ((dv + dv.permute(0,1,4,3,2))**2).sum([2,4])).sum(-1) 
                output['v_output'] = Wdz + (prob.unsqueeze(2)*s).sum(-1,keepdims=False)


            return output


    def get_sample_in_bounding_box(self, pnts,batch_size,num_points):
        
        s_mean = pnts.detach().mean(dim=0, keepdim=True)
        s_cov = pnts - s_mean
        s_cov = torch.mm(s_cov.transpose(0,1),s_cov)
        vecs = torch.eig(s_cov,True)[1].transpose(0,1)
        if torch.det(vecs) < 0:
            vecs = torch.mm(utils.get_cuda_ifavailable(torch.tensor([[0,1],[1,0]])).float(),vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1),
                (pnts.detach() - s_mean).unsqueeze(-1))
        max_x = helper[:,0,0].max().item() + 0.2
        min_x = helper[:,0,0].min().item() - 0.2
        max_y = helper[:, 1, 0].max().item() + 0.2
        min_y = helper[:, 1, 0].min().item() - 0.2
        scale_tensor = torch.tensor([[max_x - min_x, 0], [0, max_y - min_y]],device=pnts.device)
        sample = torch.rand([batch_size, pnts.shape[-1], num_points],device=pnts.device)
        helper2 = torch.bmm(scale_tensor.unsqueeze(0).repeat(batch_size, 1, 1), sample) + torch.tensor([min_x,min_y]).unsqueeze(0).unsqueeze(-1).to(pnts)

        sample = (torch.bmm(vecs.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2),helper2) + s_mean.unsqueeze(-1)).detach()
        sample = sample.permute(0,2,1)
        return sample

    def get_rand_sample(self, pnts,batch_size,num_points):
        
        rand_pnts = torch.rand([batch_size,num_points,pnts.shape[-1]],device=pnts.device)
        max_x = self.sample_box[0] 
        min_x = self.sample_box[1]
        max_y = self.sample_box[2]
        min_y = self.sample_box[3]
        if len(self.sample_box) == 3*2:
            max_z = self.sample_box[4] # 1.2
            min_z = self.sample_box[5] # -1.2
            scale_tensor = torch.tensor([[max_x - min_x, 0,0], [0,max_y - min_y,0],[0,0,max_z-min_z]],device=rand_pnts.device)
            rand_pnts = torch.bmm(scale_tensor.unsqueeze(0).repeat(rand_pnts.shape[0], 1, 1), rand_pnts.transpose(2,1)) + torch.tensor([min_x,min_y,min_z]).unsqueeze(0).unsqueeze(-1).to(rand_pnts)
        else:
            scale_tensor = torch.tensor([[max_x - min_x, 0], [0,max_y - min_y]],device=rand_pnts)
            rand_pnts = torch.bmm(scale_tensor.unsqueeze(0).repeat(rand_pnts.shape[0], 1, 1), rand_pnts.transpose(2,1)) + torch.tensor([min_x,min_y]).unsqueeze(0).unsqueeze(-1).to(rand_pnts)
        rand_pnts = rand_pnts.transpose(1,2)
        return rand_pnts



