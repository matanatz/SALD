import torch
import torch.utils.data as data
import numpy as np
import os
import logging
from utils.general import *
import logging

class DFaustDataSet(data.Dataset):

    def __init__(self,with_gt = False,with_scans=False,**kwargs):
        base_dir = kwargs['dataset_path']
        self.npyfiles_mnfld = self.get_instance_filenames(base_dir,kwargs['split'])
        self.npyfiles_dist = self.get_instance_filenames(base_dir,kwargs['split'],kwargs['dist_file_name']) if kwargs['with_dist'] else None
        self.number_of_points = kwargs['number_of_points']
        logging.debug('number of points : {0}'.format(self.number_of_points))
        self.normalization_files = self.get_instance_filenames(base_dir, kwargs['split'], '_normalization')
        self.normalization_params = [torch.from_numpy(np.expand_dims(np.load(x,allow_pickle=True).item()['center'],0)).float() for x in self.normalization_files]
        self.scans_files = self.get_instance_filenames(os.path.join(os.path.split(base_dir)[0], 'scans'),kwargs['split'], '', kwargs['scans_file_type'] if 'scans_file_type' in kwargs else 'ply') if with_scans else None
        if (with_gt):
            # Used only for evaluation
            self.gt_files = self.get_instance_filenames(os.path.join(os.path.split(base_dir)[0],'scripts'),kwargs['split'],'','obj')
            self.shapenames = [x.split('/')[-1].split('.obj')[0] for x in self.gt_files]



    def get_instance_filenames(self,base_dir,split,ext='',format='npy'):
        npyfiles = []
        l = 0
        for dataset in split:
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    j = 0
                    for shape in split[dataset][class_name][instance_name]:

                        instance_filename = os.path.join(base_dir, class_name,instance_name, shape + "{0}.{1}".format(ext,format))
                        if not os.path.isfile(instance_filename):
                            logging.error('Requested non-existent file "' + instance_filename + "' {0} , {1}".format(l,j))
                            l = l+1
                            j = j + 1
                        npyfiles.append(instance_filename)
        return npyfiles

    def __getitem__(self, index):
        
        point_set_mnlfld = torch.from_numpy(np.load(self.npyfiles_mnfld[index])).float()
        sample_non_mnfld = torch.from_numpy(np.load(self.npyfiles_dist[index])).float()
        random_idx = (torch.rand(self.number_of_points**2) * point_set_mnlfld.shape[0]).long()
        point_set_mnlfld = torch.index_select(point_set_mnlfld,0,random_idx)
        normal_set_mnfld = point_set_mnlfld[:,3:] 
        point_set_mnlfld = point_set_mnlfld[:,:3]# + self.normalization_params[index].float()

        random_idx = (torch.rand(self.number_of_points ** 2) * sample_non_mnfld.shape[0]).long()
        sample_non_mnfld = torch.index_select(sample_non_mnfld, 0, random_idx)

        return point_set_mnlfld,normal_set_mnfld,sample_non_mnfld,index

    def __len__(self):
        return len(self.npyfiles_dist)




    

   
    