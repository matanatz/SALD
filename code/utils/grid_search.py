import numpy as np
import os
import json
import submitit
import sys
import numpy as np
sys.path.append('.')
import utils.general as utils


class GridSearch:
    def __init__(self, runner, folder, exps_folder, parameters, parmas_all, conf_name, batch_size, all_exps_folder='/checkpoint/matanatz/all_exps_2', nepochs=20000, constraint_gpu=False, gpus_per_node=1, partition='learnfair',is_continue=False,cluster='waic'):
        self.cluster = cluster
        params_list = list(parameters.items())

        utils.mkdir_ifnotexists(all_exps_folder)

        if cluster == 'fb':
            executor = submitit.AutoExecutor(folder='log_test/{0}'.format(exps_folder))
            dict_params = dict(slurm_job_name=exps_folder + '_' + conf_name,
                               timeout_min=60*24,
                               gpus_per_node=gpus_per_node,
                                slurm_partition=partition,
                                nodes=1,
                                slurm_ntasks_per_node=1,
                                cpus_per_task=10)

            if constraint_gpu:
                dict_params['slurm_constraint'] = 'volta32gb'

            executor.update_parameters(**dict_params)
            self.executor = executor


            #,gpus-per-node=8,partition='dev',output='/checkpoint/%u/jobs/sample-%j.out',error='/checkpoint/%u/jobs/sample-%j.err',cpus-per-task=10)
        
        runs = []
        for i in range(len(params_list)):
            values1 = params_list[i][1]
            for v1_index, v1 in enumerate(values1):
                str = ["""include "../../{0}.conf" """.format(conf_name)]
                str = str + parmas_all
                str.append('{0} = {1}'.format(params_list[i][0],v1))
                for j in range(i+1,len(params_list)):
                    values2 = params_list[j][1]
                    
                    for v2_index,v2 in enumerate(values2):
                        str_addition = []
                        str_addition.append('{0} = {1}'.format(params_list[j][0],json.dumps(v2)))
                        run_name = '{0}_{1}_{2}_{3}_{4}'.format(conf_name,params_list[i][0],params_list[i][1][v1_index],params_list[j][0],params_list[j][1][v2_index])
                        str_addition.append('train.expname = {0}'.format(run_name))
                        print('\n'.join(str + str_addition))

                        with open(os.path.join(folder,'{0}_{1}.conf'.format(v1_index,v2_index)), "w") as text_file:
                            text_file.write('\n'.join(str + str_addition))
                        utils.mkdir_ifnotexists("{0}/exps_{1}".format(all_exps_folder,exps_folder))
                        runs.append(dict(conf=os.path.join(folder,'{0}_{1}.conf'.format(v1_index,v2_index)),
                            batch_size=batch_size,
                            nepochs=nepochs,
                            expname='',
                            gpu_index='ignore',
                            exps_folder_name="exps_{0}".format(exps_folder),
                            parallel=True,
                            workers=0,
                            is_continue=is_continue,
                            timestamp='latest',
                            checkpoint='latest',
                            debug=True,
                            quiet=False,
                            vis=True,
                            run_name=run_name))

        self.runs = runs
        self.runner = runner
    


    def run(self):
        if self.cluster == 'fb':
            jobs = self.executor.map_array(self.runner,self.runs)
            print (jobs)
        else:
            for run in self.runs:
                template = ['echo "JJJJ"',
                            'source /home/labs/waic/atzmonm/.bashrc',
                            'conda activate im_flow',
                            'cd /home/labs/waic/atzmonm/data/im_flow/code_sdf_latent_flow',
                            'python training/exp_runner.py --gpu ignore --batch_size {0} --nepoch {1} --conf {2} {3} --expsfolder {4} {5} {6} --workers {7}'.format(
                                run['batch_size'],
                                run['nepochs'],
                                run['conf'],
                                '--expname ' + run['expname'] if run['expname'] != '' else '',
                                run['exps_folder_name'],
                                '--is_continue' if run['is_continue'] else '',
                                "--parallel" if run['parallel'] else '',
                                run['workers'])]

                utils.mkdir_ifnotexists('./exp_small/runs/')

                with open("./exp_small/runs/run_{0}.sh".format(run['run_name']), "w") as text_file:
                    text_file.write('\n'.join(template))

                retries = 1
                job_command = """/shareDB/wexac_workshop/seq_arr.sh -e {1} -d ended  -c "bsub -H -J {0}[1-{1}] -env LSB_CONTAINER_IMAGE=ibdgx001:5000/gropper_1  -R "select[hname!=dgxws02]" -R "select[hname!=ibdgxws002]" -R "select[hname!=dgxws01]"  -R "select[hname!=ibdgx001]" -R "rusage[mem=32768]" -R "affinity[thread*24]" -q waic-long -app nvidia-gpu -gpu num=1:j_exclusive=yes -oo /home/labs/waic/atzmonm/runs/out_{0}.log -eo /home/labs/waic/atzmonm/runs/err_{0}.log  bash /home/labs/waic/atzmonm/data/im_flow/code_sdf_latent_flow/exp_small/runs/run_{0}.sh" """.format(
                    run['run_name'], retries)

                print(job_command)
                os.system(job_command)