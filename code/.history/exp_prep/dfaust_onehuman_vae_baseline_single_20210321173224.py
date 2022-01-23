import os
import json
import submitit
import sys
sys.path.append('../code_sdf_latent_flow')
import utils.general as utils
from exp_prep.network_training import NetworkTraining
from exp_prep.base import BaseRunner

class ExpRunner(BaseRunner):


    
    def prepare_splits_and_conf(self):
        

        output_folder_name = 'one_human'
        utils.mkdir_ifnotexists('./confs/splits/dfaust/{0}'.format(output_folder_name))
        utils.mkdir_ifnotexists('./confs/{0}'.format(output_folder_name))
        self.runs_conf = []
        self.runs = []
        
        humans = ['50002','50004','50020','50027','50022']
        humans = ['50020','50002']

        for human in humans:
            name = 'dfaust_onehuman_vae_baseline_{0}'.format(human)

            # prepare conf
            str_conf = """
            include "../dfaust_local.conf"
            train.expname = {1}
            train.network_class = model.network_linear_prob_newloss_oldsolver_int.DeformNetwork
            train.data_split = dfaust/one_human_mix/train_{2}_freq_10.json
            train.test_split = dfaust/one_human_mix/test_{2}_freq_40.json
            train.save_checkpoint_frequency = 10
            train.plot_frequency = 100
            train.auto_decoder=False
            train.debug_proj = False
            train.test_after = [20000,25000,30000,35000,37500,40000,42500,45000,47500,50000,55000,60000,65000,70000,80000,85000,90000]
            network.decoder_implicit.with_emb = False
            train.dataset.properties.number_of_points = 50
            train.dataset.properties.preload=True
            network.loss.properties.recon_loss_weight = 1.0
            network.loss.properties.grad_loss_weight = 0.1
            network.loss.properties.z_weight = 0.001
            network.loss.properties.killing_weight = 0.0
            network.loss.properties.recon_loss_weight = 1.0
            network.loss.properties.grad_loss_weight = 0.1
            network.loss.properties.z_weight = 0.0
            network.loss.properties.latent_reg_weight = 0.001
            network.v_sample_factor = 2
            network.K = 20
            network.lambda_i = 0.0
            network.with_sample = True
            network.sigma_square = 2
            network.weighted_lsq = False
            network.noise_lsq = 0.1
            network.loss.properties.prob_weight=0
            network.concate_proj=False
            network.with_sample = False
            network.shape_interpolation=2
            network.adaptive_with_sample = False
            network.adaptive_epoch = 20000
            """.format(output_folder_name,name,human)

            with open("./confs/{0}/{1}.conf".format(output_folder_name, name), "w") as text_file:
                text_file.write(str_conf)

            self.runs_conf.append(["./confs/{0}/{1}.conf".format(output_folder_name, name),name])
            self.runs.append(dict(self.params,**{'conf':"./confs/{0}/{1}.conf".format(output_folder_name, name)}))


if __name__ == "__main__":
    args = ExpRunner.get_args()
    e = ExpRunner(**dict(args._get_kwargs()))
    e.run()
    print (e.runs_conf)
