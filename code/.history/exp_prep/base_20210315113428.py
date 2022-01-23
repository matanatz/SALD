import utils.general as utils
import argparse
import submitit
import os

class BaseRunner:


    def __init__(self,
                 gpu_num=1,
                 retries=1,
                 class_trainer='training.train_implicit_can.TrainImplicitCanRunner',
                 cluster='waic',
                 gpu='auto',
                 workers=16,
                 batch_size=64,
                 cancel_parallel=False,
                 exps_folder='exps',
                 nepochs=200000,
                 is_continue=False,
                 g_queue=False):

        super().__init__()
        self.g_queue = g_queue
        self.runs_conf = []
        self.runs = []
        self.gpu_num = gpu_num
        self.retries = retries
        self.cluster = cluster
        self.class_trainer = class_trainer
        self.params = dict(batch_size=batch_size,
                        nepochs=nepochs,
                        exps_folder_name=exps_folder,
                        parallel=not cancel_parallel,
                        workers=workers,
                        expname='',
                        is_continue=is_continue,
                        timestamp='latest',
                        gpu_index =gpu, #'auto' if self.is_local else 'ignore',
                        checkpoint= 'latest',
                        debug=True,
                        quiet=False,
                        vis=True,
                        trainer=class_trainer
                           )
        if cluster == "waic":
            self.datapath = '/home/labs/waic/atzmonm/data/datasets/dfaust'
        elif cluster == 'fb':
            self.datapath = '/checkpoint/matanatz/datasets/dfaust'
        elif cluster == 'wis':
            self.datapath = '/home/atzmonm/data/datasets/dfaust'

        
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--cluster", type=str, default="waic", help="[waic,wis,fb].")
        parser.add_argument("--gpu", type=str, default="auto")
        parser.add_argument("--workers", type=int, default=32)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--retries", type=int, default=1)
        parser.add_argument("--gpu_num", type=int, default=1)
        parser.add_argument("--cancel_parallel", default=False,action="store_true")
        parser.add_argument("--is_continue", default=False,action="store_true")
        parser.add_argument("--g_queue", default=False,action="store_true")
        opt = parser.parse_args()
        return opt

    def run(self):
        
        self.prepare_splits_and_conf()

        if self.cluster == 'fb':
            from exp_prep.network_training import NetworkTraining
            executor = submitit.AutoExecutor(folder='log_test/dfaust_remove_unique_action_ad_ours_proj_v2')
            partition = 'learnfair'
            gpus_per_node = 8
            dict_params = dict(slurm_job_name='dfaust_remove_unique_action_ad_ours_proj_v2',
                               timeout_min=60 * 24 * 3,
                               gpus_per_node=gpus_per_node,
                               slurm_partition=partition,
                               nodes=1,
                               slurm_ntasks_per_node=1,
                               cpus_per_task=80,
                               slurm_mem='512GB',
                               slurm_constraint='volta32gb')
            executor.update_parameters(**dict_params)
            runner = NetworkTraining(self.class_trainer)
            jobs = executor.map_array(runner, self.runs)

            print(jobs)
        elif self.cluster == 'wis':
            for conf in self.runs_conf:
                trainrunner = utils.get_class(self.class_trainer)(
                    **dict(self.params, **{'conf': conf[0]})
                )
                trainrunner.run()
        elif self.cluster == 'waic':
            for conf in self.runs_conf:
                template = ['echo "JJJJ"',
                            'source /home/labs/waic/atzmonm/.bashrc',
                            'conda activate pytorch3d',
                            'cd /home/labs/waic/atzmonm/data/im_flow_new/im_flow/code_sdf_latent_flow',
                            'python training/exp_runner.py --gpu ignore --batch_size {0} --nepoch {1} --conf {2} {3} --expsfolder {4} {5} {6} --workers {7} --trainer {8} --checkpoint {9}'.format(self.params['batch_size'],
                                                                                                                self.params['nepochs'],
                                                                                                                conf[0],
                                                                                                                '--expname ' + self.params['expname'] if self.params['expname'] != '' else '',
                                                                                                                self.params['exps_folder_name'],
                                                                                                                '--is_continue' if self.params['is_continue'] else '',
                                                                                                                "--parallel" if self.params['parallel'] else '',
                                                                                                                self.params['workers'],
                                                                                                                self.params['trainer'],
                                                                                                                self.params['checkpoint'])]

                utils.mkdir_ifnotexists('./runs_scripts/')

                with open("./runs_scripts/run_{0}.sh".format(conf[1]), "w") as text_file:
                    text_file.write('\n'.join(template))

                retries = self.retries
                list = [""" -R "select[hname!={0}]" """.format(x) for x in  ['agn01','agn02','agn03','agn04','agn05','dgxws01','dgxws02','hgn03','hgn04','hgn05','hgn06','hgn07','hgn08','hgn09','hgn10','hgn11','hgn12','hgn13']]
                
                if self.gpu_num == 1:
                    no = ''
                    q = 'waic-long'
                else:
                    no = ''.join(list)
                    if self.g_queue:
                        q = 'gpu-short'
                    else:
                        q = 'waic-short'
                        
                job_command = """/shareDB/wexac_workshop/seq_arr.sh -e {1} -d ended  -c "bsub -H -J {0}[1-{1}]  {2}   -R "rusage[mem=65536]" -R "affinity[thread*20]" -q {3}  -gpu num={4}:j_exclusive=yes  -oo /home/labs/waic/atzmonm/runs/out_{0}_%I.log -eo /home/labs/waic/atzmonm/runs/err_{0}_%I.log  bash /home/labs/waic/atzmonm/data/im_flow_new/im_flow/code_sdf_latent_flow/runs_scripts/run_{0}.sh" """.format(conf[1],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            retries,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            no,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            q,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            self.gpu_num)

                print (job_command)
                #subprocess.run([job_command])

                os.system(job_command)
