import os
import json
import numpy as np
import sys
sys.path.append("../code")
import utils.general as utils



os.system("rm -r -f ./runs/arrayruns*")
utils.mkdir_ifnotexists('runs')

split_file = './confs/splits/shapenet/all.json'
with open(split_file, "r") as f:
    all_models = json.load(f)

str_all = ["#!/bin/csh"]
print (global_shape_index)


#length = np.array([len(l) for l in list(list(train_split.items())[0][1].items())[0][1].values()]).sum()

template = '\n'.join(["#!/bin/csh",
                      "#$ -N testarray","#$ -t {0}-{1}:1",
                      "conda activate pytorch3d",
                      #"setenv LD_LIBRARY_PATH  /home/atzmonm/data/anaconda3/lib",
                      "cd /home/atzmonm/data/im_flow/code_sdf_latent_flow/",
                      "echo before run",
            "python preprocess/preprocess_dfaust_reg_alec.py --shapeindex {2} --datapath /home/atzmonm/data/datasets/dfaust",
                      "echo after run"])


#_trainglesbaseline_betternormal


        # utils.mkdir_ifnotexists(mode)
        # utils.mkdir_ifnotexists(os.path.join(mode,name))
        # iter = length


length = global_shape_index + 1
str_all = []
for run,i in enumerate(np.array_split(np.arange(1,length+1),length/100)):
    text_file = open(os.path.join("runs/arrayruns{0}.csh".format(run)), "w")

    text_file.write(template.format( i[0] , i[-1] ,"${SGE_TASK_ID}"))
    text_file.close()
    str_all.append("""qsub -q "all2.q" -e /home/atzmonm/data/{0}/{1}_{2}.err -o /home/atzmonm/data/{0}/{1}_{2}.out ./{0}/arrayruns{1}.csh""".format('runs',run,"\$TASK_ID"))


text_file = open("./alls.csh","w")
text_file.write("""\n""".join(str_all))
text_file.close()

