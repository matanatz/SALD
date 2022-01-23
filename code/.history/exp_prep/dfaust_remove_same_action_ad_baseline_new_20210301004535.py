import os
import json
import submitit
import time
import sys
import logging
sys.path.append('../code_sdf_latent_flow')
import utils.general as utils
from exp_prep.network_training import NetworkTraining
from exp_prep.base import BaseRunner



class ExpRunner(BaseRunner):

    
    def prepare_splits_and_conf(self):
        train_sample_freq = 20
        test_sample_freq = 10

        datapath = '/checkpoint/matanatz/datasets/dfaust'
        output_folder_name = 'remove_same_action'
        utils.mkdir_ifnotexists('./confs/splits/dfaust/{0}'.format(output_folder_name))
        utils.mkdir_ifnotexists('./confs/{0}'.format(output_folder_name))
        self.runs_conf = []
        self.runs = []




        # no_action_list = ['one_leg_loose','punching','one_leg_loose','shake_hips','one_leg_jump','light_hopping_stiff'] # 'punching'
        # no_action_list = ['punching']#,'punching','one_leg_loose','shake_hips','one_leg_jump','light_hopping_stiff'] # 'punching'

        #no_action_list = ['one_leg_loose','one_leg_loose','shake_hips','one_leg_jump','light_hopping_stiff'] # 'punching'



        no_action_list = ['one_leg_loose','punching','one_leg_loose','shake_hips','one_leg_jump','light_hopping_stiff'] # 'punching'





        for no_action in no_action_list:

            name = 'dfaust_remove_same_action_no_{0}_ad_baseline'.format(no_action)


            # prepare conf
            str_conf = """
            include "../dfaust.conf"
            train.expname = {2}
            train.network_class = model.network_linear_prob_bug_2.DeformNetwork
            train.data_split = /dfaust/{0}/train_remove_same_action_no_{1}.json
            train.test_split = /dfaust/{0}/test_remove_same_action_no_{1}.json
            train.auto_decoder=True
            train.plot_corr = True
            train.save_checkpoint_frequency = 10
            train.dataset.properties.number_of_points = 50
            train.plot_frequency = 100
            train.dataset.properties.preload = False
            network.loss.properties.killing_weight = 0.001
            network.loss.properties.recon_loss_weight = 1.0
            network.K = 20
            train.latent_size = 256
            network.is_mean_shape = False
            network.loss.properties.grad_loss_weight = 0.1
            network.loss.properties.z_weight = 0.001
            network.v_noise = 0.02
            network.v_sample_factor = 2
            network.v_filter = 0.1
            network.v_projection_steps = 5
            network.lambda_i = 0.0
            network.with_sample = False
            network.sigma_square = 2
            network.weighted_lsq = False
            network.noise_lsq = 0.1
            network.loss.properties.prob_weight=0
            network.concate_proj = False
            network.proj_with_con = False
            
            """.format(output_folder_name, no_action,name)

            with open("./confs/{0}/{1}.conf".format(output_folder_name, name), "w") as text_file:
                text_file.write(str_conf)

            self.runs_conf.append(["./confs/{0}/{1}.conf".format(output_folder_name, name),name])
            self.runs.append(dict(self.params,**{'conf':"./confs/{0}/{1}.conf".format(output_folder_name, name)}))



if __name__ == "__main__":
    args = ExpRunner.get_args()
    e = ExpRunner(**dict(args._get_kwargs()))
    e.run()
    print (e.runs_conf)
