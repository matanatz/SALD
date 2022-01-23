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

        
        
        
        self.runs_conf = []
        self.runs = []




        



        exps = ['shapenet_chairs_ad','shapenet_chairs_vae','shapenet_sofas_ad','shapenet_sofas_ad','shapenet_tables_ad','shapenet_tables_vae','shapenet_planes_ad','shapenet_planes_vae','shapenet_lamps_vae','shapenet_lamps_ad'] # 'punching'





        for exp in exps:

            


            

            

            self.runs_conf.append(["./confs/{0}.conf".format(exp),exp])
            self.runs.append(dict(self.params,**{'conf':"./confs/{0}.conf".format(exp)}))



if __name__ == "__main__":
    args = ExpRunner.get_args()
    e = ExpRunner(**dict(args._get_kwargs()))
    e.run()
    print (e.runs_conf)
