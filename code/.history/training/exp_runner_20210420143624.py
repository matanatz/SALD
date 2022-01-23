import argparse
import sys
# python training/exp_runner.py --batch_size 2 --expname _uniform --workers 0 --nepoch 100000 --gpu auto --conf ./confs/dfaust_fix_latent_small_local.conf
sys.path.append("../code")

import utils.general as utils
import GPUtil
import torch


if __name__ == "__main__":
    #torch.set_num_threads(1)
    print(torch.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="Input batch size.")
    
    parser.add_argument(
        "--nepoch", type=int, default=20000, help="Number of epochs to train."
    )
    parser.add_argument("--conf", type=str, default="./confs/dfaust.conf")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--expsfolder", type=str, default="exps")
    parser.add_argument(
        "--gpu", type=str, default="ignore", help="GPU to use [default: GPU auto]."
    )
    parser.add_argument(
        "--parallel",
        default=False,
        action="store_true",
        help="If set, indicaties running on multiple gpus.",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Data loader number of workers."
    )
    parser.add_argument(
        "--is_continue",
        default=False,
        action="store_true",
        help="If set, indicates continuing from a previous run.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default='latest',
        help="The timestamp of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='latest',
        help="The checkpoint epoch number of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--debug",
        default=True,
        action="store_true",
        help="If set, debugging messages will be printed.",
    )

    parser.add_argument(
        "--cancel_vis",
        default=False,
        action="store_true",
        help="If set, cancel visualize plots in visdom.",
    )

    parser.add_argument(
        "--quiet",
        dest="quiet",
        default=False,
        action="store_true",
        help="If set, only warnings will be printed.",
    )

    parser.add_argument(
        "--trainer",
        dest="trainer",
        default='training.train_runner.TrainRunner',
        type=str
    )

    opt = parser.parse_args()



    trainrunner = utils.get_class(opt.trainer)(
        conf=opt.conf,
        batch_size=opt.batch_size,
        nepochs=opt.nepoch,
        expname=opt.expname,
        gpu_index=opt.gpu,
        exps_folder_name=opt.expsfolder,
        parallel=opt.parallel,
        workers=opt.workers,
        is_continue=opt.is_continue,
        timestamp=opt.timestamp,
        checkpoint=opt.checkpoint,
        debug=opt.debug,
        quiet=opt.quiet,
        vis=not opt.cancel_vis
    )

    trainrunner.run()
