import sys
sys.path.append('../code')
import torch
import GPUtil
from model.network_vae import DeformNetwork
import numpy as np
import os
import time
from pyhocon import ConfigFactory

if __name__ == '__main__':
    deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    gpu = deviceIDs[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
    list_nums = [80**2]

    for num_points in list_nums:

        # decoder = Decoder(latent_size=0,
        #                     dims = [ 512, 512, 512, 512,512,512,512,512],
        #                 dropout = [],
        #                 dropout_prob =  0.2,
        #                 norm_layers = [0, 1, 2, 3, 4, 5, 6, 7],
        #                 latent_in = [4],
        #                 activation = 'None',
        #                 latent_dropout = False,
        #                 weight_norm = False,
        #                   xyz_dim=3).cuda()

        conf = """
        network{
            with_vae = True
            t_samples = 1
            predict_normals_on_surfce = False
            viscosity = 0.0
            with_vae = False
            with_sample = True
            uniform_sample=True
            sample_box = [1.0,-1.0,2.0,-1.0,1.0,-1.0]
            rand_sample_factor = 1
            v_sample_factor = 4
            v_projection_steps = 5
            v_start_uniform = True
            proj_with_con = False
            v_filter = 0.0001
            v_noise = 0.02
            con_dir = False
            dir_detach=True
            dist_sample_factor = 2
            concate_proj = True
            t_include_bndry = False
            t_beta_sampling = False
            is_nearest = False
            dist_start_uniform = True
            is_mean_shape = False
        
            encoder{
                with_normals=False
            }
            decoder_implicit
            {
                dims = [ 512, 512, 512,512,512, 512, 512,512],
                dropout = []
                dropout_prob =  0.2
                norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
                latent_in = [4]
                #xyz_in_all = False
                activation = None
        
                latent_dropout = False
                weight_norm = True
                with_emb=False
                xyz_dim = 3
        
            }
            decoder_v
            {
                dims =  [512, 512, 512,512,512,512,512,512]
                dropout = []
                dropout_prob =  0.2
                norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
                latent_in = [4]
                #xyz_in_all = False
                activation = None
        
                latent_dropout = False
                weight_norm = True
                with_emb=False
                xyz_dim = 3
                is_v = True
                geometric_init = False
            }
        }
        """
        conf = ConfigFactory.parse_string(conf)
        network = DeformNetwork(conf.get_config('network'),256,False)
        network.cuda()

        sample = torch.tensor(np.random.rand(8,num_points,3)).float().cuda()
        print(torch.cuda.memory_allocated(sample.device))
        #a = network(sample.detach(),sample.detach(),sample.detach(),None,None,False, True)
        #print(torch.cuda.memory_allocated(sample.device))
        times = []
        for i in range(100):
            start = time.time()
            network(sample.detach(),sample.detach(),sample.detach(),None,None,False, False)

            end = time.time()
            times.append(end - start)
        print ("numpoints :{0} , {1}".format(num_points, np.array(times).mean()))

        # load obj
