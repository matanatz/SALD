import argparse
import sys
sys.path.append('../code_sdf_latent_flow')
import utils.general as utils
import os
import json
import plotly.express as px
import trimesh
import utils.general as utils
import logging
import torch
from pyhocon import ConfigFactory
import utils.plots as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import os
import plotly.express as px
import plotly.figure_factory as ff
import GPUtil
from torchdiffeq import odeint as odeint_normal
import imageio
from tqdm import tqdm
import base64
import tempfile
import subprocess
import select
import itertools,random
import pandas as pd
import plotly.io._orca
import retrying

def encode(tensor,name = None):
    L = tensor.shape[0]
    H = tensor.shape[1]
    W = tensor.shape[2]
    loadfile=False
    if name is None:
        loadfile=True
        t = tempfile.NamedTemporaryFile(suffix='.mp4')
        name = t.name

    command = [ 'ffmpeg',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '{}x{}'.format(W, H), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '5', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-pix_fmt', 'yuv420p',
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'h264',
        '-f', 'mp4',
        '-y', # overwrite
        name
        ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    output = bytes()

    frame = 0

    print("Encoding...")

    with tqdm(total=L) as bar:
        while frame < L:
            state = proc.poll()
            if state is not None:
                print('Could not call ffmpeg (see above)')
                raise IOError

            read_ready, write_ready, _ = select.select([proc.stdout], [proc.stdin], [])

            if proc.stdout in read_ready:
                buf = proc.stdout.read1(1024 * 1024)
                output += buf

            if proc.stdin in write_ready:
                proc.stdin.write(tensor[frame].tobytes())
                frame += 1
                bar.update()

        remaining_output, _ = proc.communicate()
        output += remaining_output
    if loadfile:
        data = open(name, 'rb').read()
        t.close()

        return data



def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class VelocityNetwork(torch.nn.Module):
        def __init__(self,first_latent,second_latent,latent_size):
            super(VelocityNetwork, self).__init__()
            self.first_latent = first_latent
            self.second_latent = second_latent
            self.latent_size = latent_size
            self.v_network = None
        def forward(self,t,y):
            middle = self.first_latent + t*(self.second_latent - self.first_latent)
            if self.latent_size > 0:
                y = torch.cat([y,middle],dim=-1)
            return torch.bmm(self.v_network(y,None,False,cat_latent=False)[0].reshape(y.shape[0],3,-1),(self.second_latent - self.first_latent).T.unsqueeze(0).repeat(y.shape[0],1,1)).squeeze()
            #return self.v_network(y,None,False,cat_latent=False)[0]

class ODEfunc(torch.nn.Module):
    def __init__(self, diffeq):
        super(ODEfunc, self).__init__()
        self.diffeq = diffeq
        self.register_buffer("_num_evals", torch.tensor(0.))

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def forward(self, t, states):
        y = states[0]
        t = torch.ones(y.size(0), 1).to(y) * t.clone().detach().requires_grad_(True).type_as(y)
        self._num_evals += 1
        for state in states:
            state.requires_grad_(True)

        # Sample and fix the noise.
        if self._e is None:
            self._e = torch.randn_like(y, requires_grad=True).to(y)

        with torch.set_grad_enabled(True):
            if len(states) == 1:  # unconditional CNF
                dy = self.diffeq(t, y)
                return (dy,)
            else:
                assert 0, "`len(states)` should be 2 or 3"

def evaluate_with_load(gpu, parallel, split_test, conf, exps_folder_name, name_override, timestamp, checkpoint, with_opt, resolution, all_couples,with_optimize_line=False,fix_coup=False,length=10,save_animation=True,no_k=False,retry=False,cancel_ffmpeg=False):
    if gpu != 'ignore':
        if gpu == "auto":
            deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                        excludeUUID=[])
            gpu = deviceIDs[0]
        else:
            gpu = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    conf = ConfigFactory.parse_file(conf)
    conf['train']['dataset']['properties']['preload'] = False
    splitfilename = './confs/splits/'
    if (split_test):
        splitfilename += conf.get_string('train.test_split')
    else:
        splitfilename += conf.get_string('train.data_split')

    with open(splitfilename, "r") as f:
        split = json.load(f)

    ds = utils.get_class(conf.get_string('train.dataset.class'))(split=split, **conf.get_config('train.dataset.properties'))
    
    logging.info ("total ds : {0}".format(len(ds)))

    if name_override != '':
        exp_name = args.override    
    else:
        exp_name = conf.get_string('train.expname')


    if timestamp == 'latest':
        timestamps = os.listdir(os.path.join(conf.get_string('train.base_path'),exps_folder_name,exp_name))
        timestamp = sorted(timestamps)[-1]
    elif timestamp == 'find':
        timestamps = [x for x in os.listdir(os.path.join('../',exps_folder_name,exp_name))
                      if not os.path.isfile(os.path.join('../',exps_folder_name,exp_name,x))]
        for t in timestamps:
            cpts = os.listdir(os.path.join('../',exps_folder_name,exp_name,t,'checkpoints/ModelParameters'))

            for c in cpts:
                if args.epoch + '.pth' == c:
                    timestamp = t
    else:
        timestamp = timestamp

    base_dir = os.path.join(conf.get_string('train.base_path'), exps_folder_name, exp_name, timestamp)
    if (gpu == 'cpu'):
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"),map_location=torch.device('cpu'))
    else:
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"))
        if conf.get_bool('train.auto_decoder') and not 'test' in splitfilename:
            if os.path.isfile(os.path.join(base_dir,'checkpoints', "LatentCodes", checkpoint + '.pth')):
                data = torch.load(os.path.join(base_dir,'checkpoints', "LatentCodes", checkpoint + '.pth'))
            
                    # if not self.lat_vecs[0].size()[1] == data["latent_codes"].size()[2]:
                    #     raise Exception("latent code dimensionality mismatch")

                    # for i in range(len(self.lat_vecs)):
                    #     self.lat_vecs[i] = data["latent_codes"][i].cuda()
                lat_vecs = torch.nn.Embedding(len(ds), conf.get_int('train.latent_size'), max_norm=1.0)
                lat_vecs.load_state_dict(data['latent_codes'])
                lat_vecs = utils.get_cuda_ifavailable(lat_vecs)
            else:
                lat_vecs = None
            # if conf.get_string('network.sample_method') == 'vae':
            #     # data = torch.load(os.path.join(base_dir,'checkpoints', "LatentCodes", args.checkpoint + '_sigma.pth'))
            #     # lat_vecs_sigma = torch.nn.Embedding(len(ds), conf.get_int('train.latent_size'), max_norm=1.0)
            #     # lat_vecs_sigma.load_state_dict(data['latent_codes'])
            #     # lat_vecs_sigma = utils.get_cuda_ifavailable(lat_vecs_sigma)
            #     lat_vecs_sigma = None
            # else:
            lat_vecs_sigma = None

        else:
            lat_vecs_sigma = None
            lat_vecs = None
    logging.info ('loaded model')
    saved_model_epoch = saved_model_state["epoch"]

    network = utils.get_class(conf.get_string('train.network_class'))(conf=conf.get_config('network'),latent_size=conf.get_int('train.latent_size'),auto_decoder=conf.get_int('train.auto_decoder'))

    if (parallel):
        network.load_state_dict(
            {'.'.join(k.split('.')[1:]): v for k, v in saved_model_state["model_state_dict"].items() if not 'temp' in k})
    else:
        network.load_state_dict(saved_model_state["model_state_dict"])

    network = utils.get_cuda_ifavailable(network)
    evaluate(network=network,
             exps_folder_name=exps_folder_name,
             experiment_name=exp_name,
             timestamp=timestamp,
             ds=ds,
             epoch=saved_model_epoch,
             with_opt=with_opt,
             resolution=resolution,
             with_normals=False,
             conf=conf,
             index=-1,
             chamfer_only=False,
             recon_only=False,
             lat_vecs=lat_vecs,
             lat_vecs_sigma=lat_vecs_sigma,
             split_filename=splitfilename,
             all_couples=all_couples,with_optimize_line=with_optimize_line,fix_coup=fix_coup,length=length,no_k=no_k,save_animation=save_animation,retry=retry,cancel_ffmpeg=cancel_ffmpeg)

def evaluate(network,exps_folder_name, experiment_name, timestamp, ds, epoch, with_opt, resolution, with_normals, conf, index, chamfer_only, recon_only, lat_vecs, lat_vecs_sigma,with_ode=False,with_cuts=False,with_animation=True,visdomer=None,env='',step_log=None, split_filename = None, with_video = False,reuse_window=None,z_func=None,with_optimize_line=False,all_couples = 1,fix_coup=False,length=10,no_k=False,save_animation=True,retry=False,cancel_ffmpeg=False):
    if type(network) == torch.nn.parallel.DataParallel:
        old_vnoise = network.module.v_noise
    else:
        old_vnoise = network.v_noise
    network.v_noise = 0.0
    utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_int'))
    if not split_filename is None:
        utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_int', split_filename.split('/')[-1].split('.json')[0]))
        path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_int', split_filename.split('/')[-1].split('.json')[0], str(epoch))
        env = '/'.join([env,'evaluation_int', split_filename.split('/')[-1].split('.json')[0], str(epoch)])
    else:
        path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_int', str(epoch))

        env = '/'.join([env,'evaluation_int', str(epoch)])
    utils.mkdir_ifnotexists(path)
    if not step_log is None:
        trace_steploss = []
        selected_stepdata = step_log        
        for x in selected_stepdata.columns:
            if 'loss' in x:
                trace_steploss.append(go.Scatter(x=np.arange(len(selected_stepdata)),y=selected_stepdata[x],mode='lines',name=x,visible='legendonly'))

        fig = go.Figure(data=trace_steploss)
        
        visdomer.plot_plotly(fig, env=env)
    plot_cmpr = False

    
    counter = 0
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)
    dataloader_all = torch.utils.data.DataLoader(ds,
                                                  batch_size=min(4,len(ds)),
                                                  shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)

    data_iter = iter(dataloader)
    data_all_iter = iter(dataloader_all)
    first_pc, first_normals,sample_nonmnfld, first_idx = next(data_iter)
    first_color = ds.get_colors(first_pc[0])
    # for i in range(np.random.randint(1,10)):
    #     second_pc, second_normals, second_idx = next(data_iter)
    #second_pc, second_normals, second_idx = next(data_iter)
    #second_pc, second_normals, second_idx = next(data_iter)

    couples = [[0,1],
    [0,2],
    [0,3]]
    # [1,2],
    # [1,3],
    # [2,3]]
    np.random.seed(1)# if all_couples != 1 else None

    if fix_coup:
        couples = [[5,7],[7,2],[2,4] , [4,0]]
        couples = [[7788,6496]]
    else:
        g = itertools.combinations(range(len(ds)),2)
        alist = list(g)
        np.random.shuffle(alist)
        if all_couples > -1:
            couples = alist[:all_couples]
        else:
            couples = alist[:10]
            #couples = [[np.random.randint(len(ds)),np.random.randint(len(ds))] for i in range(all_couples)]
            #couples = [[0,1],[0,2],[0,3]]
        #couples = [[0,1],[1,2]]
    first = True

    camera = dict(
        up=dict(x= -0.010351348985487011, y= 0.0067430768332676465, z= 0.9999236873326891),
        center=dict(x=0, y=0.0, z=0),
        eye=dict(x= 0.7449130410645354, y= 1.1889344144068383, z= 0.1266091702498)
    )

    camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0.0, z=0),
    eye=dict(x=0, y=0.6, z=0.9)

    )
    for c in couples:
        first_pc = utils.get_cuda_ifavailable(ds[c[0]][0]).unsqueeze(0)
        first_normals = utils.get_cuda_ifavailable(ds[c[0]][1]).unsqueeze(0)
        first_idx = utils.get_cuda_ifavailable(torch.tensor(ds[c[0]][3])).unsqueeze(0)
        second_pc = utils.get_cuda_ifavailable(ds[c[1]][0]).unsqueeze(0)
        second_normals = utils.get_cuda_ifavailable(ds[c[1]][1]).unsqueeze(0)
        second_idx = utils.get_cuda_ifavailable(torch.tensor(ds[c[1]][3])).unsqueeze(0)

        if (conf.get_bool('train.auto_decoder')):
            first_latent = lat_vecs[first_idx] if type(lat_vecs) == torch.Tensor else lat_vecs(first_idx)
            second_latent = lat_vecs[second_idx] if type(lat_vecs) == torch.Tensor else lat_vecs(second_idx)
        else:
            _,first_latent,_ = network(manifold_points=first_pc, manifold_normals=first_normals,sample_nonmnfld=None, latent=None, latent_sigma_inputs=None, only_encoder_forward=True, only_decoder_forward=False)
            _,second_latent,_ = network(manifold_points=second_pc, manifold_normals=second_normals,sample_nonmnfld=None, latent=None,latent_sigma_inputs=None, only_encoder_forward=True, only_decoder_forward=False)
            #network(first_pc,only_encoder_forward=True)

        traces = []
        arrows = []
        points = []
        layout = go.Layout(width=1200, height=1200, scene=dict(camera=camera ,xaxis=dict(range=[-2,2], autorange=False,showbackground=False,visible=False),
                                                                    yaxis=dict(range=[-2, 2], autorange=False,showbackground=False,visible=False),
                                                                    zaxis=dict(range=[-2, 2], autorange=False,showbackground=False,visible=False),
                                                                    aspectratio=dict(x=1, y=1, z=1)))

        traces = [go.Scatter3d(
                    x=first_pc[0][:,0].detach().cpu(),
                    y=first_pc[0][:,1].detach().cpu(),
                    z=first_pc[0][:,2].detach().cpu(),
                    name='first',
                    mode='markers',
                    marker=dict(
                        size=5,
                        line=dict(
                            width=1,
                        ),
                        opacity=1.0,
                        showscale=True,
                        color=first_color)),
                    go.Scatter3d(
                    x=second_pc[0,:,0].detach().cpu(),
                    y=second_pc[0,:,1].detach().cpu(),
                    z=second_pc[0,:,2].detach().cpu(),
                    name='second',
                    mode='markers',
                    marker=dict(
                        size=5,
                        line=dict(
                            width=1,
                        ),
                        opacity=1.0,
                        showscale=True
                    ))]
        traces_anim = traces
        traces_prob = traces
        if with_ode:
            intervals = np.linspace(0,1,100)
            diffeq = VelocityNetwork(first_latent, second_latent,network.v_network.latent_size)
            diffeq.v_network = network.v_network
            odefunc = ODEfunc(
                        diffeq=diffeq,
                    )

            res = network(manifold_points=first_pc, manifold_normals=None, latent=first_latent, latent_sigma_inputs=None, only_encoder_forward=False, only_decoder_forward=False, spider_head=None, is_additional_latent=False)
            
            start_mesh = plt.get_surface_trace(first_pc[0],
                                            network,
                                            first_latent,
                                            64,
                                            0,
                                            False,
                                            True,
                                            True,
                                            is_3d=True,
                                            name='i_{0}'.format(i),
                                            z_func=z_func)['mesh_export']

            solution = torch.tensor(start_mesh.vertices).float().to(first_pc)
            first_vertex_color = ds.get_colors(solution).cpu().detach().numpy()

            verts = start_mesh.vertices
            def tri_indices(simplices):
                    return ([triplet[c] for triplet in simplices] for c in range(3))

            I, J, K = tri_indices(start_mesh.faces)
            traces_ode = []
            traces_ode.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, 
                                intensity=first_vertex_color.astype(np.float)/5.0,colorscale=[[0, 'gold'],
                        [0.5, 'mediumturquoise'],
                        [1, 'magenta']],name='first', showlegend=True, showscale=False, opacity=1.0))
            
            for j in tqdm(range(len(intervals) - 1)):
                integration_times = torch.tensor([intervals[j], intervals[j+1]], requires_grad=False)

                odefunc.before_odeint()
                odeint = odeint_normal

                atol = 1e-6
                rtol = 1e-6
                test_solver = 'dopri5'
                test_atol = atol
                test_rtol = rtol
                res = []
                for pnts in torch.split(solution, 10000, dim=0):
                    states = (pnts,)

                    state_t = odeint(
                                odefunc,
                                states,
                                integration_times,
                                atol=test_atol,
                                rtol=test_rtol,
                                method=test_solver,
                            )

                    solution = state_t[0][1]

                # project solution to zero level set of f
                    curr_projection = solution.detach().unsqueeze(0)
                    latent = first_latent + intervals[j+1]*(second_latent - first_latent)
                    for i in range(5):
                        
                        network_eval, grad,_ = network.implicit_map(curr_projection.detach(), latent, True)

                        sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True)
                        curr_projection = curr_projection - (network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)))
                    res.append(curr_projection.detach().squeeze(0))

                solution = torch.cat(res,0)
                verts = solution.cpu().detach().numpy()
                traces_ode.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=I, j=J, k=K, 
                                    intensity=first_vertex_color.astype(np.float)/5.0,colorscale=[[0, 'gold'],
                            [0.5, 'mediumturquoise'],
                            [1, 'magenta']],name='{0}'.format(j),showscale=False, opacity=1.0))                            

            fig = go.Figure(
            data=[traces_ode[0]],
            layout=go.Layout(
                width=1200, height=1200, scene=dict(camera=camera,xaxis=dict(range=[-2, 2], autorange=False,visible=False),
                                                                    yaxis=dict(range=[-2, 2], autorange=False,visible=False),
                                                                    zaxis=dict(range=[-2, 2], autorange=False,visible=False),
                                                                    aspectratio=dict(x=1, y=1, z=1)),
                title="Start Title",
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                method="animate",
                                args=[None,{"frame": {"duration": 10}}]),
                                {
                                "args": [[None], {"frame": {"duration": 0},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                                }])]
            ),
            frames=[go.Frame(data=[x]) for x in traces_ode])
            filename = '{0}/animation_ode_{1}_{2}.html'.format(path, c[0], c[1])
            logging.info('saving in {0}'.format(filename))
            offline.plot(fig,
                    filename=filename,
                    auto_open=False)
            
            # traces.append(go.Scatter3d(
            #         x=solution[:,0].detach().cpu(),
            #         y=solution[:,1].detach().cpu(),
            #         z=solution[:,2].detach().cpu(),
            #         name='first_to_second',
            #         mode='markers',
            #         marker=dict(
            #             size=5,
            #             line=dict(
            #                 width=1,
            #             ),
            #             opacity=1.0,
            #             showscale=True,
            #             color=first_color,
            #         )))
                    
        # filename = '{0}/corr.html'.format(path)
        # fig1 = go.Figure(data=traces,layout=layout)
        # offline.plot(fig1, filename=filename, auto_open=False)
        
        if with_optimize_line:
            t_vals = utils.get_cuda_ifavailable(torch.arange(1,length+1)).unsqueeze(-1)/float(length + 1)
            t_vals = t_vals #* torch.norm(second_latent - first_latent,p=2,dim=-1,keepdim=True)
            # temp = t_vals[-1]
            # t_vals[1] = t_vals[-1]
            # t_vals[-1] = temp
            first_latent = first_latent.detach()
            second_latent = second_latent.detach()
            latent_points = first_latent + t_vals*(second_latent - first_latent)
            latent_points = latent_points.detach()
            latent_points.requires_grad_(True)
            lr = 1.0e-3
            optimizer = torch.optim.Adam([latent_points], lr=lr)
            num_iterations = 800

            #network.with_sample=False
            for e in range(num_iterations):
                latent_points_bndry = torch.cat([first_latent, latent_points, second_latent], dim=0)
                dirs = latent_points[1:] - latent_points[:-1]
                dirs = torch.cat([latent_points[0] - first_latent ,dirs,second_latent - latent_points[-1]],dim=0)
                lengths = torch.norm(dirs,dim=-1,p=2)
                segement = torch.distributions.categorical.Categorical(probs=lengths).sample([8])
                print (segement)
                t = torch.rand([segement.shape[0]]).to(first_latent)

                z_sampels = latent_points_bndry[segement] + t.unsqueeze(-1) * dirs[segement]
                sample_latent_size = z_sampels.shape[0]
                curr_projection = network.get_rand_sample(first_pc, sample_latent_size, first_pc.shape[1]).detach()
                proj_latent = z_sampels.detach()


                # Find bounding box
                for i in range(1):
                    network_eval, grad, _ = network.implicit_map(curr_projection.detach(), proj_latent, True)
                    network_eval = network_eval.detach()
                    grad = grad.detach()
                    sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True).detach()
                    curr_projection = curr_projection - (network_eval.abs() > 1.0e-4).repeat(1, 1, 3) * (
                                network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)))

                rand_pnts = torch.rand([sample_latent_size, curr_projection.shape[1], curr_projection.shape[-1]]).to(
                    curr_projection)
                center = curr_projection.mean(1, keepdims=True)
                curr_projection = curr_projection - center
                max_bnd = curr_projection.topk(k=3, dim=1)[0].mean(1) + 0.05
                min_bnd = curr_projection.topk(k=3, dim=1, largest=False)[0].mean(1) - 0.05
                max_min = max_bnd - min_bnd
                rand_pnts = torch.bmm(torch.diag_embed(max_min), rand_pnts.transpose(1, 2)) + min_bnd.unsqueeze(-1)
                curr_projection = rand_pnts.transpose(1, 2).detach() + center
                lr = [0.1, 0.25, 0.5, 1.0, 1.0]
                for i in range(5):
                    network_eval, grad, _ = network.implicit_map(curr_projection.detach(), proj_latent, True)
                    network_eval = network_eval.detach()
                    grad = grad.detach()
                    sum_square_grad = torch.sum(grad ** 2, dim=-1, keepdim=True).detach()
                    curr_projection = curr_projection - lr[i] * (network_eval.abs() > network.v_filter).repeat(1, 1, 3) * (
                                network_eval * (grad / sum_square_grad.clamp_min(1.0e-6)))

                network_eval, proj_grad, input_con = network.implicit_map(curr_projection.detach(), proj_latent, True,
                                                                       grad_with_repsect_to_latent=True)
                v_filter = (network_eval.abs() < network.v_filter).detach()

                curr_projection = (curr_projection + network.v_noise * torch.randn_like(curr_projection)).detach()
                curr_projection.requires_grad_(True)
                v_output = network.v_network(curr_projection, z_sampels, False)[0]
                v_output = v_output.reshape(v_output.shape[0], v_output.shape[1], -1, max(network.v_latent_size, 1))


                eval_proj, proj_grad, input_con = network.implicit_map(curr_projection, z_sampels, True,
                                                                    grad_with_repsect_to_latent=True)
                dfdz = proj_grad[:, :, -network.latent_size:]
                proj_grad = proj_grad[:, :, :first_pc.shape[-1]]
                grad_filter = (proj_grad ** 2).sum(-1, keepdim=True) > 1.0e-1
                v_hat = - torch.einsum('bpk,bpj->bpkj', proj_grad, dfdz) / (proj_grad ** 2).sum(-1,keepdim=True).unsqueeze(-1)
                s = v_hat + v_output - (torch.einsum('bpkj,bpkj->bpj', v_output,
                                                     proj_grad.unsqueeze(-1).repeat(1, 1, 1, v_output.shape[-1])) / (
                                                    proj_grad ** 2).sum(-1, keepdims=True)).unsqueeze(
                    2) * proj_grad.unsqueeze(-1).repeat(1, 1, 1, v_output.shape[-1])




                u = dirs[segement].unsqueeze(1).repeat(1, v_output.shape[1], 1)
                dv = torch.cat([torch.autograd.grad(s[:, :, i, :],
                                                    curr_projection,
                                                    u,
                                                    retain_graph=True,
                                                    create_graph=True)[0].unsqueeze(2) for i in
                                range(v_output.shape[2])], dim=2)
                # Killing vector field term

                # Killing vector field term
                killing = dv + dv.permute(0,1,3,2)

                loss = ((v_filter.unsqueeze(-1).detach()*grad_filter.unsqueeze(-1).detach()*(killing**2)).sum([-1,-2]) * torch.norm(dirs[segement],dim=-1,keepdim=True)) .mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logging.info("iteration : {0} , loss {1}".format(e, loss.item()))
                

        int_start = len(traces)
        v_con = []

        #t_vals = np.linspace(0,1,length+2)
        #t_vals = np.concatenate([np.linspace(0, 0.5, (length + 2) // 2), np.linspace(0.5, 1.0, (length + 2) // 2)])

        t_vals = np.concatenate([np.linspace(0, 0.5, (length + 2) // 2), np.linspace(0.5, 1.0, (length + 2) // 2)[1:]])     
        t_vals = np.linspace(-0,1,length+2)

        def decoder_killing_linear(x,latent,u):
            
            res = network(x,None,x,latent,None,False,False,outside_sample=x,outside_dir=u)
            if 'dv' in res :
                return res['dv'],res['clusters'],res['v_output']#,res['residual']
            elif 'dv_prob' in res:
                return res['dv_prob'],res['clusters'],res['v_output']#,res['residual']
            else:
                return None
               
        if type(network) == torch.nn.parallel.DataParallel:
            if hasattr(network.module,'shape_interpolation'):
                shape_interpolation = network.module.shape_interpolation
            else:
                shape_interpolation = None
            if hasattr(network.module,'is_mean_shape'):
                is_mean_shape = network.is_mean_shape
            else:
                is_mean_shape = None
            
        else:
            if hasattr(network,'shape_interpolation'):
                shape_interpolation = network.shape_interpolation
            else:
                shape_interpolation = None
            if hasattr(network,'is_mean_shape'):
                is_mean_shape = network.is_mean_shape
            else:
                is_mean_shape = None

        for i,t in enumerate(t_vals):
            if with_optimize_line:
                if i == 0:
                    middle = first_latent
                elif i==len(t_vals) - 1:
                    middle = second_latent
                else:
                    middle = latent_points[i - 1:i]
            else:
                if shape_interpolation is None:
                    if not is_mean_shape:
                        middle = first_latent + t*(second_latent - first_latent)
                    else:
                        if t < 0.5:
                            middle = (1- 2*t)*first_latent

                        else:
                            middle = (2*t - 1)*second_latent
                else:
                    if shape_interpolation == 0:
                        if t < 0.5:
                            middle = (1- 2*t)*first_latent
                        else:
                            middle = (2*t - 1)*second_latent
                    elif shape_interpolation == 1:
                        middle = first_latent + t*(second_latent - first_latent)
                    elif shape_interpolation == 2:
                        first_normalize = torch.nn.functional.normalize(first_latent,p=2,dim=-1)
                        second_normalize = torch.nn.functional.normalize(second_latent,p=2,dim=-1)
                        Omega = torch.acos((first_normalize * second_normalize).sum(-1,keepdim=True))
                        norm_int = torch.norm(first_latent,p=2,dim=-1,keepdim=True) + t * (torch.norm(second_latent,p=2,dim=-1,keepdim=True) - torch.norm(first_latent,p=2,dim=-1,keepdim=True))

                        P_t = (torch.sin((1 - t) * Omega) / torch.sin(Omega)) * first_normalize + (torch.sin((t) * Omega) / torch.sin(Omega)) * second_normalize
                        middle = norm_int * P_t
            # res = network(first_pc, manifold_normals=None, latent= middle, latent_sigma_inputs=None, only_encoder_forward=False, only_decoder_forward=False, spider_head=None, is_additional_latent=False)
            # v_con.append((res['v_con']**2).sum(-1).mean().item())
            first_surface = plt.get_surface_trace(None,
                                            network,
                                            middle,
                                            128,
                                            0,
                                            True,
                                            True,
                                            True,
                                            is_3d=True,
                                            name='i_{0}'.format(i),
                                            z_func=z_func)
            if not first_surface['mesh_export'] is None:
                mesh = first_surface['mesh_export']
                # components = mesh.split(only_watertight=False)
                # areas = np.array([c.area for c in components], dtype=np.float)
                # mesh = components[areas.argmax()]

                #torch.max(torch.min(torch.tensor(mesh.vertices).float(),
                #                                                                 torch.tensor([[1,1.7,1]]).float().repeat(mesh.vertices.shape[0],1)),
                #                                                torch.tensor([[-1,-0.7,-1]]).float().repeat(mesh.vertices.shape[0],1))
                second_surface = plt.get_surface_trace(points=torch.tensor(mesh.vertices).float(),
                                                decoder=network,
                                                latent=middle,
                                                resolution=resolution,
                                                mc_value=0,
                                                is_uniform=False,
                                                verbose=True,
                                                save_ply=True,
                                                is_3d=True,
                                                name='i_{0}'.format(i),
                                                z_func=z_func)

                filename = '{0}/int_{1}_{2}_{3}.ply'.format(path,c[0],c[1],i)
                second_surface['mesh_export'].export(filename,'ply')
                
                verts = utils.get_cuda_ifavailable(torch.tensor(second_surface['mesh_export'].vertices).float())
                #con = decoder_consistency(verts.unsqueeze(0),middle).squeeze().cpu().detach().numpy()
                # killing_random_perpnt = decoder_killing(verts.unsqueeze(0),middle,True,True).squeeze().cpu().detach().numpy()
                # killing_random = decoder_killing(verts.unsqueeze(0),middle,True,False).squeeze().cpu().detach().numpy()
                killing =[]
                all_prob = []
                #[killing.append([]) for i in range(1)]
                # cond = []
                # [cond.append([]) for i in range(1)]
                # hess = []
                v_output = []
                #residual = []
                verts_sample = utils.get_cuda_ifavailable(torch.tensor(trimesh.sample.sample_surface(second_surface['mesh_export'],10000)[0]).float())
                print ("no k {0} ".format(no_k))
                if not no_k:
                    for pnts in torch.split(verts_sample,verts_sample.shape[0], dim=0):
                        d = decoder_killing_linear(pnts.unsqueeze(0),middle,torch.torch.nn.functional.normalize(middle, p=2, dim=-1))
                        if d is None:
                            no_k = True
                            break
                        else:
                            dv,prob,v = d
                        killing.append(dv.detach())
                        all_prob.append(prob.detach())
                        v_output.append(v.detach())
                        #residual.append(res)


                        #[kk.append(ss) for kk,ss in zip(killing,s)]

                    
                        #[c.append(rr) for c,rr in zip(cond,r)]
                        #hess.append(t.squeeze().cpu().detach().numpy())

                #killing = [np.concatenate(kk, axis=0) for kk in killing]
                # cond = [np.concatenate(cc,axis=0) for cc in cond]
                # hess = np.concatenate(hess,axis=0)
                #logging.info("killing : {0}".format(killing.shape))
                #normal_v = decoder_normal_v(verts.unsqueeze(0),middle,second_latent - first_latent).squeeze().cpu().detach().numpy()
                if not no_k:
                    killing = torch.cat(killing,dim=1).detach()
                    filename = '{0}/killingloss_{1}_{2}_{3}.csv'.format(path, c[0],c[1],i)
                    pd.DataFrame({'killing':killing.cpu().detach().numpy()[0]}).to_csv(filename)

                    prob = torch.cat(all_prob,dim=1).detach()
                    v_output = torch.cat(v_output,dim=1).detach()
                    #residual = torch.cat(residual,dim=1).detach()
                    #prob = torch.nn.Softmax(dim=-1)(torch.cat(all_prob,dim=1).detach())
                    kk = (killing**2).sum([-1,-2]) if len(killing.shape)>2 else killing
                    kk_mean = kk.mean()
                    kk_std = kk.std()
                    kk_max = kk.max()
                    kk_min = kk.min()
                
                def tri_indices(simplices):
                    return ([triplet[c] for triplet in simplices] for c in range(3))

                I, J, K = tri_indices(second_surface['mesh_export'].faces)
                verts = verts.detach().cpu().numpy()
                anim_trace = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],hoverinfo='text',
                                    i=I, j=J, k=K, 
                                    name='i_{0}'.format(i),
                                    color='#ffffff',
                                    opacity=1.0, flatshading=False,
                                    lighting=dict(diffuse=1, ambient=0, specular=0), lightposition=dict(x=0, y=0, z=-1),
                                    showlegend=True, showscale=True)
                # second_trace = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],hoverinfo='text',
                #                     i=I, j=J, k=K,
                #                          color='#ffffff',
                #                          opacity=1.0, flatshading=False,
                #                          lighting=dict(diffuse=1, ambient=0, specular=0),
                #                          lightposition=dict(x=0, y=0, z=-1))
                                #         intensity=con.astype(np.float),colorscale=[[0, 'gold'],
                                # [0.01, 'mediumturquoise'],
                                # [0.1, 'magenta']],name='i_{0}_consistency'.format(i), showlegend=True, showscale=True, opacity=1.0)
                if not no_k:
                    # killing_latent_dir = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    #                     hoverinfo='text',
                    #                     hovertemplate="x: %{x} <br> y: %{y} <br> z: %{z} <br> loss: %{text} <extra> " + "mean {0} <br> std {1} <br> max {2}".format(kk_mean,kk_std,kk_max) + "</extra>",
                    #                     text=kk[0].cpu().detach().numpy(),
                    #                     i=I, j=J, k=K,
                    #                         intensity=kk[0].cpu().detach().numpy().astype(np.float),colorscale=[[0, 'gold'],
                    #                 [0.01, 'mediumturquoise'],
                    #                 [0.1, 'magenta']],name='i_{0}_killing'.format(i), showlegend=True, showscale=True, opacity=1.0)]
                                    # go.Cone(
                                    # x=verts[:, 0],
                                    # y=verts[:, 1],
                                    # z=verts[:, 2],
                                    # u=v_output[0, :, 0].cpu().detach().numpy(),
                                    # v=v_output[0, :, 1].cpu().detach(),
                                    # w=v_output[0, :, 2].cpu().detach(),
                                    # sizemode="scaled",
                                    # sizeref=20,
                                    # anchor="tail",
                                    # showlegend=True,
                                    # showscale=True,
                                    # name='i_{0}_v'.format(i)) ]
                    
                    probs = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                        hoverinfo='text',
                                        text=prob[0,:,j].cpu().numpy().astype(np.float),
                                        i=I, j=J, k=K,
                                            intensity=prob[0,:,j].cpu().numpy().astype(np.float),colorscale=[[0, 'gold'],
                                    [0.5, 'mediumturquoise'],
                                    [1.0, 'magenta']],name='prob_{0}'.format(j), showlegend=True, showscale=True, opacity=1.0) for j in range(min(prob.shape[-1],20))]
                    probs_max = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                        hoverinfo='text',
                                        text=prob[0,:].cpu().numpy().argmax(-1).astype(np.float),
                                        i=I, j=J, k=K,
                                            intensity=prob[0,:].cpu().numpy().argmax(-1).astype(np.float),colorscale=px.colors.qualitative.Set1,name='prob_max_{0}'.format(i), showlegend=True, showscale=True, opacity=1.0) ] 
                            # [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            #             hoverinfo='text',
                            #             text=residual[0,:,j].cpu().numpy().astype(np.float),
                            #             i=I, j=J, k=K,
                            #                 intensity=residual[0,:,j].cpu().numpy().astype(np.float),colorscale=[[0, 'gold'],
                            #         [0.5, 'mediumturquoise'],
                            #         [1.0, 'magenta']],name='residual_{0}'.format(j), showlegend=True, showscale=True, opacity=1.0) for j in range(residual.shape[-1])]
                    traces = traces + probs_max #+  [hess_latent_dir]#,killing_random,killing_random_perpnt]
                    if i % 5 == 0:
                        traces_prob = traces_prob + probs
                #
                # killing_latent_dir = killing_latent_dir + [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                #                                 hoverinfo='text',
                #                                 text=kk,
                #                                 i=I, j=J, k=K,
                #                                 intensity=kk.astype(np.float),colorscale=[[0, 'gold'],
                #                                                                                   [0.01, 'mediumturquoise'],
                #                                                                                   [0.1, 'magenta']],name='i_{0}_killing_e-{1}'.format(i,j), showlegend=True, showscale=True, opacity=1.0) for j,kk  in enumerate(killing)]

                # cond_latent_dir = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                #                                hoverinfo='text',
                #                                text=np.log(cc),
                #                                i=I, j=J, k=K,
                #                                intensity=np.log(cc).astype(np.float),colorscale=[[0, 'gold'],
                #                                                                               [0.01, 'mediumturquoise'],
                #                                                                               [0.1, 'magenta']],name='i_{0}_cond_e-{1}'.format(i,j), showlegend=True, showscale=True, opacity=1.0) for j,cc in enumerate(cond)]
                # hess_latent_dir = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                #                         hoverinfo='text',
                #                         text=np.log(hess),
                #                         i=I, j=J, k=K,
                #                         intensity=np.log(hess).astype(np.float),colorscale=[[0, 'gold'],
                #                                                                     [0.01, 'mediumturquoise'],
                #                                                                     [0.1, 'magenta']],name='i_{0}_hess'.format(i), showlegend=True, showscale=True, opacity=1.0)
                #
                # normal_v_latent_dir = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                #                                hoverinfo='text',
                #                                text=normal_v,
                #                                i=I, j=J, k=K,
                #                                intensity=normal_v.astype(np.float), colorscale=[[0, 'gold'],
                #                                                                                [0.01,
                #                                                                                 'mediumturquoise'],
                #                                                                                [0.1, 'magenta']],
                #                                name='i_{0}_normal_v'.format(i), showlegend=True,
                #                                showscale=True, opacity=1.0)
                # killing_random = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],hoverinfo='text',
                #                     i=I, j=J, k=K,
                #                     text=killing_random, 
                #                         intensity=killing_random.astype(np.float),colorscale=[[0, 'gold'],
                #                 [0.01, 'mediumturquoise'],
                #                 [0.1, 'magenta']],name='i_{0}_killing_random'.format(i), showlegend=True, showscale=True, opacity=1.0)
                # killing_random_perpnt = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],hoverinfo='text',
                #                     i=I, j=J, k=K, 
                #                     text=killing_random_perpnt,
                #                         intensity=killing_random_perpnt.astype(np.float),colorscale=[[0, 'gold'],
                #                 [0.01, 'mediumturquoise'],
                #                 [0.1, 'magenta']],name='i_{0}_killing_random_perpnt'.format(i), showlegend=True, showscale=True, opacity=1.0)
                
                    
                traces_anim = traces_anim + [anim_trace]
            if with_cuts and i % 4 == 0:
                if type(network) == torch.nn.parallel.DataParallel:
                    pnts = network.module.get_rand_sample(first_pc,1,100)
                else:
                    pnts = network.get_rand_sample(first_pc,1,100)
                def decoder_sdf(x,latent):
                    if type(network) == torch.nn.parallel.DataParallel:
                        eval_proj, proj_grad,input_con,_ = network.module.implicit_map(x,latent, True)
                    else:                    
                        eval_proj, proj_grad,input_con,_ = network.implicit_map(x,latent, True)
                    return eval_proj.squeeze(-1)

                plt.plot_cuts(first_pc[0],decoder_sdf,path,'sdf',i,False,middle,4,lambda x: x)#np.sign(x)*(-1.0/10.0)*np.log(1+np.abs(x))
                #plt.plot_cuts(first_pc[0],decoder_consistency,path,'con',i,False,middle,4)
                #plt.plot_cuts(first_pc[0],decoder_v,path,'v_norm',i,False,middle,4)

        
        fig = go.Figure(data=[go.Scatter(x=t_vals,y=np.array(v_con),mode='lines',name='v_con')])
        # filename = '{0}/con_line_{1}_{2}.html'.format(path, c[0], c[1])
        # logging.info('saving in {0}'.format(filename))
        # offline.plot(fig,
        #             filename=filename,
        #             auto_open=False)
        if len(traces) > 0:
            if with_animation and len(traces_anim) > 2:
                fig = go.Figure(
                    data=[traces_anim[int_start]],
                    layout=go.Layout(
                        width=1200, height=1200, scene=dict(camera=camera,xaxis=dict(range=[-2, 2], autorange=False,showbackground=False,visible=False),
                                                                            yaxis=dict(range=[-2, 2], autorange=False,showbackground=False,visible=False),
                                                                            zaxis=dict(range=[-2, 2], autorange=False,showbackground=False,visible=False),
                                                                            aspectratio=dict(x=1, y=1, z=1)),
                        title="Start Title",
                        updatemenus=[dict(
                            type="buttons",
                            buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None,{"frame": {"duration": 60, "redraw": True},"transition":{"easing": "linear"}}]),
                                        {"args": [[None], {"frame": {"duration": 0},
                                                        "mode": "immediate",
                                                        "transition": {"duration": 0}}],
                                        "label": "Pause",
                                        "method": "animate"
                                        }])]
                    ),
                    frames=[go.Frame(data=[x]) for x in traces_anim[int_start:]])
                filename = '{0}/animation_{1}_{2}_{3}_{4}.html'.format(path, index, c[0],c[1],with_optimize_line)
                logging.info('saving in {0}'.format(filename))
                if save_animation:
                    offline.plot(fig,
                                filename=filename,
                                auto_open=False)
                
                if not cancel_ffmpeg:
                    images = []
                    if retry:
                        unwrapped = plotly.io._orca.request_image_with_retrying.__wrapped__
                        wrapped = retrying.retry(wait_random_min=1000)(unwrapped)
                        plotly.io._orca.request_image_with_retrying = wrapped

                    for s,t in enumerate(traces_anim[int_start:]):
                        fig = go.Figure(data=[t],layout=layout)
                        try:
                            error = False
                            fig.write_image('{0}/{1}.png'.format(path,s))
                        except Exception as e:
                            logging.error (str(e))
                            error = True
                        if not error:
                            images.append(np.expand_dims(imageio.imread('{0}/{1}.png'.format(path,s)),0))
                    #visdomer.plot_video(np.concatenate(images,0),env=env)
                    if not error:
                        try:
                            data = encode(np.concatenate(images,0)[:,:,:,:3],name='{0}/animation_{1}_{2}_{3}.mp4'.format(path, index, c[0],c[1]))
                        except Exception as e:
                            logging.error (str(e))
        
            #visdomer.plot_plotly(fig,env=env)
            if not visdomer is None and with_video:
                images = []
                for s,t in enumerate(traces[int_start:]):
                    fig = go.Figure(data=[t],layout=layout)
                    fig.write_image('{0}/{1}.png'.format(path,s))
                    images.append(np.expand_dims(imageio.imread('{0}/{1}.png'.format(path,s)),0))
                #visdomer.plot_video(np.concatenate(images,0),env=env)
                data = encode(np.concatenate(images,0)[:,:,:,:3])
                videodata = """
                <video controls>iklerbinutibrnvfgunhtffrjfgkhetk
                    <source type="video/mp4" src="data:video/mp4;base64,{}">
                    Your browser does not support the video tag.
                </video>
                """.format(base64.b64encode(data).decode('utf-8'))
                visdomer.plot_txt(videodata, opts=dict(title='Sequence {}'.format(i)), env=env)
                    



            fig = go.Figure()
            fig.update_layout(layout)
            [fig.add_trace(t) for t in traces]

            filename = '{0}/killing_{1}_{2}_{3}_{4}.html'.format(path, index, c[0], c[1],with_optimize_line)
            logging.info('saving in {0}'.format(filename))
            offline.plot(fig,
                        filename=filename,
                        auto_open=False)

            if not visdomer is None:
                visdomer.plot_plotly(fig,env=env)

            fig = go.Figure()
            fig.update_layout(layout)
            [fig.add_trace(t) for t in traces_prob]

            filename = '{0}/prob_{1}_{2}_{3}_{4}.html'.format(path, index, c[0], c[1],with_optimize_line)
            logging.info('saving in {0}'.format(filename))
            # offline.plot(fig,
            #             filename=filename,
            #             auto_open=False)

        if first:
            first = False
            # Eval using latent sample
            if (conf.get_bool('train.auto_decoder')):
                #latent = lat_vecs(torch.arange(lat_vecs.num_embeddings).to(first_pc).long())
                #latent_sigma = lat_vecs_sigma(torch.arange(lat_vecs_sigma.num_embeddings).to(first_pc).long()) if not lat_vecs_sigma is None else 0.1 * torch.ones_like(latent)
                additional_latent = conf.get_float('train.sigma') * torch.randn_like(first_latent.repeat(4,1))
                sigma=None
            else:
                pc, normals,_, idx = next(data_all_iter)
                pc = utils.get_cuda_ifavailable(pc)
                normals = utils.get_cuda_ifavailable(normals)
                additional_latent,latent,sigma = network(manifold_points=pc, manifold_normals=normals,sample_nonmnfld=None, latent=None, latent_sigma_inputs = None, only_encoder_forward=True, only_decoder_forward=False)

            
            
            def plot_latent(latent_to_plot,sigma,name):
                fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}],
                                                        [{"type": "scene"}, {"type": "scene"}]])
                dict_layout = dict(xaxis=dict(range=[-2.0, 2.0], autorange=False),
                                                            yaxis=dict(range=[-2.0, 2.0], autorange=False),
                                                            zaxis=dict(range=[-2.0, 2.0], autorange=False),
                                                            aspectratio=dict(x=1, y=1, z=1))
                fig.layout.scene.update(dict_layout)
                fig.layout.scene2.update(dict_layout)
                fig.layout.scene3.update(dict_layout)
                fig.layout.scene4.update(dict_layout)
                for i,latent in enumerate(latent_to_plot):
                    trace = plt.get_surface_trace(None,
                                                        network,
                                                        latent,
                                                        resolution*4,
                                                        0,
                                                        True,
                                                        True,
                                                        True,
                                                        is_3d=True,
                                                        name='{0}'.format(latent.data),
                                                        z_func=z_func)['mesh_trace']
                    if len(trace)>0:
                        fig.add_trace(trace[0],row=i//2 + 1,col=i%2 + 1)
                div = offline.plot(fig, include_plotlyjs=False, output_type='div', auto_open=False)
                if not visdomer is None:
                    visdomer.plot_plotly(fig,env=env)
                div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
                js = '''
                                                                    <script>
                                                                    var gd = document.getElementById('{div_id}');
                                                                    var isUnderRelayout = false
                    
                                                                    gd.on('plotly_relayout', () => {{
                                                                    console.log('relayout', isUnderRelayout)
                                                                    if (!isUnderRelayout) {{
                                                                            Plotly.relayout(gd, 'scene2.camera', gd.layout.scene.camera)
                                                                            .then(() => {{ isUnderRelayout = false }}  )
                                                                            Plotly.relayout(gd, 'scene3.camera', gd.layout.scene.camera)
                                                                            .then(() => {{ isUnderRelayout = false }}  )
                                                                            Plotly.relayout(gd, 'scene4.camera', gd.layout.scene.camera)
                                                                            .then(() => {{ isUnderRelayout = false }}  )
                                                                        }}
                    
                                                                    isUnderRelayout = true;
                                                                    }})
                                                                    </script>'''.format(div_id=div_id)
                div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
                with open(os.path.join(path, "compare_{0}_{1}_{2}.html".format(name,c[0],c[1])),"w") as text_file:
                    text_file.write(div)

            plot_latent(additional_latent,sigma,"rand")
        # plot_latent(latent,"train")

        if type(network) == torch.nn.parallel.DataParallel:
            network.module.v_noise = old_vnoise
        else:
            network.v_noise = old_vnoise

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--expname", required=False, help='The experiment name to be evaluated.',default='')
    arg_parser.add_argument("--override", required=False, help='Override exp name.',default='')
    arg_parser.add_argument("--exps_folder_name", default="exps", help='The experiments directory.')
    arg_parser.add_argument("--timestamp", required=False, default='latest')
    arg_parser.add_argument("--conf", required=False , default='./confs/dfaust_local.conf')
    arg_parser.add_argument("--checkpoint", help="The checkpoint to test.", default='latest')
    arg_parser.add_argument("--split_test", default=False, action="store_true", required=False,help="The split to evaluate.")
    arg_parser.add_argument("--parallel", default=False, action="store_true", help="Should be set to True if the loaded model was trained in parallel mode")
    arg_parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto].')
    arg_parser.add_argument('--with_opt', default=False, action="store_true", help='If set, optimizing latent with reconstruction Loss versus input scan')
    arg_parser.add_argument('--resolution', default=128, type=int, help='Grid resolution')
    arg_parser.add_argument('--index', default=-1, type=int, help='Grid resolution')
    arg_parser.add_argument('--chamfer_only', default=False,action="store_true")
    arg_parser.add_argument('--recon_only', default=True,action="store_true")
    arg_parser.add_argument('--with_optimize_line', default=False, action="store_true")
    arg_parser.add_argument('--fix_coup', default=False, action="store_true")
    arg_parser.add_argument('--length', default=10, type=int, help='Seq length')
    arg_parser.add_argument('--num_c', default=20, type=int, help='Seq length')
    arg_parser.add_argument('--no_k', default=False, action="store_true")
    arg_parser.add_argument('--cancel_save_animation', default=False, action="store_true")
    arg_parser.add_argument('--cancel_ffmpeg', default=False, action="store_true")
    arg_parser.add_argument('--retry', default=False, action="store_true")



    args = arg_parser.parse_args()
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logging.info ("running")
    
    
    lat_vecs_sigma = None
    
    
    evaluate_with_load(gpu=args.gpu,
                       parallel=args.parallel,
                       split_test=args.split_test,
                       conf=args.conf,
                       exps_folder_name=args.exps_folder_name,
                       name_override=args.override,
                       timestamp=args.timestamp,
                       checkpoint=args.checkpoint,
                       with_opt=args.with_opt,
                       resolution=args.resolution,
                       all_couples=args.num_c,
                       with_optimize_line=args.with_optimize_line,
                       fix_coup=args.fix_coup,
                       length=args.length,
                       no_k=args.no_k,
                       save_animation=not args.cancel_save_animation,
                       retry=args.retry,
                       cancel_ffmpeg=args.cancel_ffmpeg)

    # evaluate(
    #     network=utils.get_cuda_ifavailable(network),
    #     exps_folder_name=args.exps_folder_name,
    #     experiment_name= name,
    #     timestamp=timestamp,
    #     ds=ds,
    #     epoch=saved_model_epoch,
    #     with_opt=args.with_opt,
    #     resolution=args.resolution,
    #     with_normals=conf.get_bool('network.encoder.with_normals'),
    #     conf=conf,
    #     index=args.index,
    #     chamfer_only=args.chamfer_only,
    #     recon_only=args.recon_only,
    #     lat_vecs=lat_vecs,
    #     lat_vecs_sigma=lat_vecs_sigma,
    #     with_ode=False,
    #     with_cuts=True,
    #     with_animation=True,
    #     z_func=None,
    #     with_optimize_line=False,
    #     all_couples = True
    # )


