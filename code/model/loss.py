from torch import nn
import torch
import utils.general as utils
import numpy as np
import logging


class GenLoss(nn.Module):
    def __init__(self,):
        super().__init__()


class SALDLoss(GenLoss):
    def __init__(self,**kwargs):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()
        self.recon_loss_weight = kwargs['recon_loss_weight']
        self.grad_loss_weight = kwargs['grad_loss_weight']
        self.z_weight = kwargs['z_weight']
        self.grad_on_surface_weight = kwargs['grad_on_surface_weight']
        self.latent_reg_weight = kwargs['latent_reg_weight']

        logging.debug("""recon_loss_weight : {0} 
                         grad_loss_weight : {1}
                         z_weight : {2}
                         grad_on_surface_weight : {3}
                         latent_reg_weight : {4}
                         """.format(self.recon_loss_weight,
                                                           self.grad_loss_weight,
                                                           self.z_weight,
                                                           self.grad_on_surface_weight,
                                                           self.latent_reg_weight,
                                                           ))


    def forward(self, network_outputs,normals_gt,normals_nonmnfld_gt,pnts_mnfld, gt_nonmnfld,epoch):
        debug = {}
        recon_loss = self.l1_loss(network_outputs['non_mnfld_pred'].abs().squeeze(),gt_nonmnfld)
        debug['recon_loss'] = recon_loss

        
        loss = self.recon_loss_weight*recon_loss

        if 'grad_on_surface' in network_outputs.keys() and self.grad_on_surface_weight > 0:
            grad_loss =  torch.min(torch.abs(network_outputs['grad_on_surface'].squeeze() - normals_gt).sum(-1),
                                                             torch.abs(network_outputs['grad_on_surface'].squeeze() + normals_gt).sum(-1)).mean()
            debug['grad_on_surface_loss'] = grad_loss

            loss = loss + self.grad_on_surface_weight * grad_loss

        if 'non_mnfld_pred_grad' in network_outputs.keys() and self.grad_loss_weight > 0:
            grad_loss = torch.min(torch.abs(network_outputs['non_mnfld_pred_grad'].squeeze() - normals_nonmnfld_gt).sum(-1),
                                    torch.abs(network_outputs['non_mnfld_pred_grad'].squeeze() + normals_nonmnfld_gt).sum(-1)).mean()

            loss = loss + self.grad_loss_weight * grad_loss
            debug['grad_loss'] = grad_loss

        if 'latent_reg' in network_outputs.keys() and self.latent_reg_weight > 0:
            loss = loss + self.latent_reg_weight * network_outputs['latent_reg'].mean()
            debug['latent_reg_loss'] = network_outputs['latent_reg'].mean()
        
        if 'norm_square_latent' in network_outputs.keys() and self.z_weight > 0:
            loss = loss + self.z_weight * network_outputs['norm_square_latent'].mean()
            debug['z_loss'] = network_outputs['norm_square_latent'].mean()
        
        debug['total_loss'] = loss
        return {"loss": loss,"loss_monitor":debug}