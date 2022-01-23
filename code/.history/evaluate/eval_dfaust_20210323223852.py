import argparse
import sys
sys.path.append('../code_sdf_latent_flow')
import utils.general as utils
import os
import json
import trimesh
import utils.general as utils
import point_cloud_utils as pcu
import logging
from datasets.datasets import DFaustDataSet
import torch
from pyhocon import ConfigFactory
import utils.plots as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import os
import GPUtil
import pandas as pd
from tqdm import tqdm
def adjust_learning_rate(
        initial_lr, optimizer, num_iterations, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def optimize_latent(conf, latent, ds, itemindex, network,lat_vecs):
    latent.detach_()
    latent.requires_grad_()
    lr = 1.0e-3
    optimizer = torch.optim.Adam([latent], lr=lr)

    #**conf.get_config('network.loss.properties')
    loss_func = utils.get_class(conf.get_string('network.loss.loss_type'))(
        recon_loss_weight=1,grad_on_surface_weight=0,grad_loss_weight=0.1,z_weight=0.001,dist_loss_weight=0,killing_weight=0.00005,v_con_weight=0.0,v_velocity_weight=0.0,latent_reg_weight=0.0,viscosity=0.0,normal_v_weight=0.0,logsumexp=False,alpha=0.0,p_dv=2,dfdz_weight=0,latent_eiko_grad_weight=0.1,v_grad_weight=0.0,prob_weight=0.0)

    loss_func = utils.get_class(conf.get_string('network.loss.loss_type'))(
        recon_loss_weight=1,grad_on_surface_weight=0,grad_loss_weight=0.1,z_weight=0.001,dist_loss_weight=0,killing_weight=0.0,v_con_weight=0.0,v_velocity_weight=0.0,latent_reg_weight=0.0,viscosity=0.0,normal_v_weight=0.0,logsumexp=False,alpha=0.0,p_dv=2,dfdz_weight=0,latent_eiko_grad_weight=0.0,v_grad_weight=0.0,prob_weight=0.0)

    num_iterations = 800

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)
    network.with_sample = False
    network.adaptive_with_sample = False
    idx_latent = utils.get_cuda_ifavailable(torch.arange(lat_vecs.num_embeddings))
    for e in range(num_iterations):
        #network.with_sample = e > 100
        pnts_mnfld,normals_mnfld,sample_nonmnfld,indices = ds[itemindex]
        
        pnts_mnfld = utils.get_cuda_ifavailable(pnts_mnfld).unsqueeze(0)
        normals_mnfld = utils.get_cuda_ifavailable(normals_mnfld).unsqueeze(0)
        sample_nonmnfld = utils.get_cuda_ifavailable(sample_nonmnfld).unsqueeze(0)
        indices = utils.get_cuda_ifavailable(indices).unsqueeze(0)
        outside_latent = lat_vecs(idx_latent[np.random.choice(np.arange(lat_vecs.num_embeddings),4,False)])

        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        outputs = network(pnts_mnfld, None, sample_nonmnfld[:,:,:3], latent, None, False, only_decoder_forward=False, spider_head=None, is_additional_latent=False)
        #loss_res = self.loss(network_outputs=outputs, normals_gt=normals_mnfld, normals_nonmnfld_gt = sample_nonmnfld[:,:,3:6], pnts_mnfld=pnts_mnfld, gt_nonmnfld=sample_nonmnfld[:,:,-1],epoch=epoch)
        loss_res = loss_func(network_outputs=outputs, normals_gt=normals_mnfld, normals_nonmnfld_gt = sample_nonmnfld[:,:,3:6], pnts_mnfld=pnts_mnfld, gt_nonmnfld=sample_nonmnfld[:,:,-1],epoch=-1)
        loss = loss_res["loss"]
        #loss = outputs['dv_prob'].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logging.info("iteration : {0} , loss {1}".format(e, loss.item()))
        logging.info("mean {0} , std {1}".format(latent.mean().item(), latent.std().item()))

    
    #network.with_sample = True
    return latent



def evaluate_with_load(gpu, conf, exps_folder_name, override, timestamp, checkpoint ,parallel, resolution, chamfer_only=False, recon_only=False, plot_cmpr=True,eval_train=False,is_action=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info("running")
    conf = ConfigFactory.parse_file(conf)

    if override != '':
        expname = override    
    else:
        expname = conf.get_string('train.expname') 

    if timestamp == 'latest':
        timestamps = os.listdir(os.path.join(conf.get_string('train.base_path'),exps_folder_name, expname))
        timestamp = sorted(timestamps)[-1]
    elif timestamp == 'find':
        timestamps = [x for x in os.listdir(os.path.join('../',exps_folder_name,expname))
                      if not os.path.isfile(os.path.join('../',exps_folder_name,expname,x))]
        for t in timestamps:
            cpts = os.listdir(os.path.join('../',exps_folder_name,expname,t,'checkpoints/ModelParameters'))

            for c in cpts:
                if args.epoch + '.pth' == c:
                    timestamp = t
    else:
        timestamp = timestamp
    
    base_dir = os.path.join(conf.get_string('train.base_path'),exps_folder_name, expname, timestamp)
    if (gpu == 'cpu'):
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"),map_location=torch.device('cpu'))
    else:
        saved_model_state = torch.load(os.path.join(base_dir, 'checkpoints', 'ModelParameters', checkpoint + ".pth"))
    logging.info('loaded model')
    saved_model_epoch = saved_model_state["epoch"]

    network = utils.get_class(conf.get_string('train.network_class'))(conf=conf.get_config('network'),latent_size=conf.get_int('train.latent_size'),auto_decoder=conf.get_int('train.auto_decoder'))

    if (parallel):
        network.load_state_dict(
            {'.'.join(k.split('.')[1:]): v for k, v in saved_model_state["model_state_dict"].items()})
    else:
        network.load_state_dict(saved_model_state["model_state_dict"])

    if conf.get_bool('train.auto_decoder') :
        split_filename = './confs/splits/{0}'.format(conf.get_string('train.data_split'))
        with open(split_filename, "r") as f:
            split = json.load(f)

        ds = utils.get_class(conf.get_string('train.dataset.class'))(split=split, with_gt=True,
                                                                     **conf.get_config('train.dataset.properties'))
        total_files = len(ds)

        lat_vecs = torch.nn.Embedding(total_files, conf.get_int('train.latent_size'), max_norm=1.0)
        if os.path.isfile(os.path.join(base_dir,'checkpoints', "LatentCodes", checkpoint + '.pth')):
            data = torch.load(os.path.join(base_dir,'checkpoints', "LatentCodes",  checkpoint + ".pth"))
            lat_vecs.load_state_dict(data['latent_codes'])
            lat_vecs = utils.get_cuda_ifavailable(lat_vecs)
        else:
            lat_vecs = None
    else:
        lat_vecs = None

    
    #def evaluate(network,exps_folder_name, experiment_name, timestamp, epoch, resolution, conf, index, chamfer_only, recon_only,lat_vecs):
    evaluate(
        network=utils.get_cuda_ifavailable(network),
        exps_folder_name=exps_folder_name,
        experiment_name=expname,
        timestamp=timestamp,
        epoch=saved_model_epoch,
        resolution=resolution,
        conf=conf,
        index=-1,
        chamfer_only=chamfer_only,
        recon_only=recon_only,
        lat_vecs=lat_vecs,
        plot_cmpr=plot_cmpr,
        with_gt=True,
        is_action=is_action
    )


def evaluate(network,exps_folder_name, experiment_name, timestamp, epoch, resolution, conf, index, chamfer_only, recon_only,lat_vecs,plot_cmpr=False,with_gt=False,is_action=False):

    if type(network) == torch.nn.parallel.DataParallel:
        network = network.module
        
    with_opt = True
    chamfer_results = dict(files=[],reg_to_gen_chamfer=[],reg_to_gen_coverage=[],gen_to_reg_chamfer=[],gen_to_reg_coverage=[],scan_to_gen_chamfer=[],scan_to_gen_coverage=[],gen_to_scan_chamfer=[],gen_to_scan_coverage=[],sinkhorn_dist_reg=[],sinkhorn_dist_scan=[])

    
    split_filename = './confs/splits/{0}'.format(conf.get_string('train.test_action_split' if is_action else 'train.test_split' ))
    with open(split_filename, "r") as f:
        split = json.load(f)
    
    ds = utils.get_class(conf.get_string('train.dataset.class'))(split=split,with_gt=with_gt,with_scans=True,scans_file_type='ply',
                                                                **conf.get_config('train.dataset.properties'))
    total_files = len(ds)
    logging.info ("total files : {0}".format(total_files))
    prop = conf.get_config('train.dataset.properties')
    prop['number_of_points'] = int(np.sqrt(30000))
    ds_eval_scan = utils.get_class(conf.get_string('train.dataset.class'))(split=split,with_gt=True,
                                                                **prop)

    prop['number_of_points'] = int(np.sqrt(1000))
    ds_sink_eval_scan = utils.get_class(conf.get_string('train.dataset.class'))(split=split,with_gt=True,
                                                                **prop)
    
    utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_sink'))
    utils.mkdir_ifnotexists(os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_sink', split_filename.split('/')[-1].split('.json')[0]))
    path = os.path.join(conf.get_string('train.base_path'), exps_folder_name, experiment_name, timestamp, 'evaluation_sink', split_filename.split('/')[-1].split('.json')[0], str(epoch))

    utils.mkdir_ifnotexists(path)

    counter = 0
    dataloader = torch.utils.data.DataLoader(ds,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0, drop_last=False, pin_memory=True)

    names = ['_'.join([ds.npyfiles_dist[i].split('/')[-3:][0],ds.npyfiles_dist[i].split('/')[-3:][2]]).split('_dist_triangle.npy')[0] for i in range(len(ds.npyfiles_dist))]
    # for j,n in enumerate(names):
    #     if n == '50022_punching.002678':
    #         index = j
    i = 1
    # index = index + 1
    for data in tqdm(dataloader):
        if ((index == -1 or index == i )):
            logging.info (counter)
            #logging.info (ds.npyfiles_mnfld[data[-1].item()].split('/'))
            counter = counter + 1

            [logging.debug("evaluating " + ds.npyfiles_mnfld[data[-1][i]]) for i in range(len(data[-1]))]

            input_pc = utils.get_cuda_ifavailable(data[0])
            input_normal = utils.get_cuda_ifavailable(data[1])
            filename = ['{0}/nonuniform_iteration_{1}_{2}_id.ply'.format(path, epoch, ds.npyfiles_mnfld[data[-1][i].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1][i].item()].split('/')[-1].split('.npy')[0]) for i in range(len(data[-1]))][0]

            if conf.get_bool('train.auto_decoder'):
                if not os.path.isfile(filename) and lat_vecs is None:
                    if with_opt:
                        latent = utils.get_cuda_ifavailable(torch.zeros([1, conf.get_int('train.latent_size')]))
                        latent = latent + 1e-5*torch.randn_like(latent)
                        latent = optimize_latent(conf,
                                                latent,
                                                ds,
                                                data[-1],
                                                network,
                                                lat_vecs)
                    else:
                        latent = lat_vecs(utils.get_cuda_ifavailable(data[-1]))
                        
                else:
                    latent = None
            else:
                _,latent,_ = network(manifold_points=input_pc,
                                manifold_normals=input_normal,
                                latent=None,
                                latent_sigma_inputs = None,
                                only_encoder_forward=True,
                                only_decoder_forward=False)
            pnts_to_plot = input_pc

            
            if chamfer_only:
                if (os.path.isfile(filename)):
                    reconstruction = trimesh.load(filename)
                    logging.info ('loaded : {0}'.format(filename))
            else:
                if (os.path.isfile(filename)):
                    reconstruction = trimesh.load(filename)
                    logging.info ('loaded : {0}'.format(filename))
                else:
                    if not latent is None:
                        reconstruction = plt.plot_surface(with_points=False,
                                        points=pnts_to_plot.detach()[0],
                                        decoder=network,
                                        latent=latent,
                                        path=path,
                                        epoch=epoch,
                                        in_epoch=ds.npyfiles_mnfld[data[-1].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1].item()].split('/')[-1].split('.npy')[0],
                                        shapefile=ds.npyfiles_mnfld[data[-1].item()],
                                        resolution=resolution,
                                        mc_value=0,
                                        is_uniform_grid=False,
                                        verbose=True,
                                        save_html=False,
                                        save_ply=True,
                                        overwrite=True,
                                        is_3d=True,
                                        z_func={'id':lambda x:x})
            if reconstruction is None and not latent is None:
                i = i + 1
                continue

            if not recon_only:
                # if (with_opt):
                    
                #     recon_after_latentopt = optimize_latent(latent, ds, data[-1], network, path, epoch,resolution)

                recon_after_latentopt = reconstruction
                normalization_params_filename = ds.normalization_files[data[-1]]
                logging.debug("normalization params are " + normalization_params_filename)
                    
                normalization_params = np.load(normalization_params_filename,allow_pickle=True)
                scale = normalization_params.item()['scale']
                center = normalization_params.item()['center']

                if with_gt:
                    gt_mesh_filename = ds.gt_files[data[-1]]
                    ground_truth_points = trimesh.Trimesh(trimesh.sample.sample_surface(utils.as_mesh(trimesh.load(gt_mesh_filename)), 30000)[0])
                    dists_to_reg = utils.compute_trimesh_chamfer(
                        gt_points=ground_truth_points,
                        gen_mesh=reconstruction,
                        offset=-center,
                        scale=1./scale,
                    )

                    gen_pnts = trimesh.sample.sample_surface(reconstruction,1000)[0]

                    a = gen_pnts
                    b = trimesh.sample.sample_surface(utils.as_mesh(trimesh.load(gt_mesh_filename)), 1000)[0]
                    w_a = np.ones(a.shape[0])
                    w_b = np.ones(b.shape[0])
                    M = pcu.pairwise_distances(a, b)
                    P = pcu.sinkhorn(w_a, w_b, M, eps=1e-3)
                    sinkhorn_dist_reg = (M*P).sum() 
                    print (sinkhorn_dist_reg)


                dists_to_scan = utils.compute_trimesh_chamfer(
                    gt_points=trimesh.Trimesh(ds_eval_scan[data[-1]][0].cpu().numpy()),
                    gen_mesh=reconstruction,
                    offset=0,
                    scale=1.,
                )

                gen_pnts = trimesh.sample.sample_surface(reconstruction,961)[0]

                a = gen_pnts
                b = ds_sink_eval_scan[data[-1]][0].cpu().numpy()
                w_a = np.ones(a.shape[0])
                w_b = np.ones(b.shape[0])
                M = pcu.pairwise_distances(a, b)
                P = pcu.sinkhorn(w_a, w_b, M, eps=1e-3)
                sinkhorn_dist_scan = (M*P).sum() 
                print (sinkhorn_dist_scan)

                if (False):
                    chamfer_dist_after_opt = utils.compute_trimesh_chamfer(
                        gt_points=ground_truth_points,
                        gen_mesh=recon_after_latentopt,
                        offset=-center,
                        scale=1. / scale,
                    )

                    chamfer_dist_scan_after_opt = utils.compute_trimesh_chamfer(
                        gt_points=trimesh.Trimesh(input_pc[0].cpu().numpy()),
                        gen_mesh=recon_after_latentopt,
                        offset=0,
                        scale=1.,
                        one_side=True
                    )

                    chamfer_results.append(
                        (
                            ds.gt_files[data[-1]],
                            chamfer_dist,
                            chamfer_dist_scan,
                            chamfer_dist_after_opt,
                            chamfer_dist_scan_after_opt
                        )
                    )
                else:
                    if with_gt:
                        chamfer_results['files'].append(ds.gt_files[data[-1]])
                        chamfer_results['reg_to_gen_chamfer'].append(dists_to_reg['gt_to_gen_chamfer'])
                        chamfer_results['reg_to_gen_coverage'].append(dists_to_reg['gt_to_gen_coverage'])
                        chamfer_results['gen_to_reg_chamfer'].append(dists_to_reg['gen_to_gt_chamfer'])
                        chamfer_results['gen_to_reg_coverage'].append(dists_to_reg['gen_to_gt_coverage'])

                    chamfer_results['scan_to_gen_chamfer'].append(dists_to_scan['gt_to_gen_chamfer'])
                    chamfer_results['scan_to_gen_coverage'].append(dists_to_scan['gt_to_gen_coverage'])
                    chamfer_results['gen_to_scan_chamfer'].append(dists_to_scan['gen_to_gt_chamfer'])
                    chamfer_results['gen_to_scan_coverage'].append(dists_to_scan['gen_to_gt_coverage'])
                    chamfer_results['sinkhorn_dist_reg'].append(sinkhorn_dist_reg)
                    chamfer_results['sinkhorn_dist_scan'].append(sinkhorn_dist_scan)


                    

                if (plot_cmpr  and i % 1 == 0):
                    if (with_opt):
                        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}],
                                                                [{"type": "scene"}, {"type": "scene"}]],
                                            subplot_titles=["Input", "Registration",
                                                            "Ours", "Ours after opt"])

                    else:
                        fig = make_subplots(rows=1, cols=2 + int(with_gt), specs=[[{"type": "scene"}] * (2 + int(with_gt))],
                                            subplot_titles=("input pc", "Ours","Registration") if with_gt else ("input pc", "Ours"))

                    fig.layout.scene.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    fig.layout.scene2.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
                    if with_gt:
                        fig.layout.scene3.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    aspectratio=dict(x=1, y=1, z=1)))
                    if (with_opt):
                        fig.layout.scene4.update(dict(xaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    yaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    zaxis=dict(range=[-1.5, 1.5], autorange=False),
                                                    aspectratio=dict(x=1, y=1, z=1)))

                    scan_mesh = utils.as_mesh(trimesh.load(ds.scans_files[data[-1]]))

                    scan_mesh.vertices = (scan_mesh.vertices - center)/scale

                    def tri_indices(simplices):
                        return ([triplet[c] for triplet in simplices] for c in range(3))

                    I, J, K = tri_indices(scan_mesh.faces)
                    color = '#ffffff'
                    trace = go.Mesh3d(x=scan_mesh.vertices[:, 0], y=scan_mesh.vertices[:, 1],
                                    z=scan_mesh.vertices[:, 2],
                                    i=I, j=J, k=K, name='scan',
                                    color=color, opacity=1.0, flatshading=False,
                                    lighting=dict(diffuse=1, ambient=0, specular=0), lightposition=dict(x=0, y=0, z=-1))
                    fig.add_trace(trace, row=1, col=1)


                    I, J, K = tri_indices(reconstruction.faces)
                    color = '#ffffff'
                    trace = go.Mesh3d(x=reconstruction.vertices[:, 0], y=reconstruction.vertices[:, 1], z=reconstruction.vertices[:, 2],
                                        i=I, j=J, k=K, name='our',
                                        color=color, opacity=1.0,flatshading=False,lighting=dict(diffuse=1,ambient=0,specular=0),lightposition=dict(x=0,y=0,z=-1))
                    if (with_opt):
                        fig.add_trace(trace, row=2, col=1)

                        I, J, K = tri_indices(recon_after_latentopt.faces)
                        color = '#ffffff'
                        trace = go.Mesh3d(x=recon_after_latentopt.vertices[:, 0], y=recon_after_latentopt.vertices[:, 1],
                                        z=recon_after_latentopt.vertices[:, 2],
                                        i=I, j=J, k=K, name='our_after_opt',
                                        color=color, opacity=1.0, flatshading=False,
                                        lighting=dict(diffuse=1, ambient=0, specular=0),
                                        lightposition=dict(x=0, y=0, z=-1))
                        fig.add_trace(trace, row=2, col=2)
                    else:
                        fig.add_trace(trace,row=1,col=2)

                    if with_gt:
                        gtmesh = utils.as_mesh(trimesh.load(gt_mesh_filename))
                        gtmesh.vertices = (gtmesh.vertices - center)/scale
                        I, J, K = tri_indices(gtmesh.faces)
                        trace = go.Mesh3d(x=gtmesh.vertices[:, 0], y=gtmesh.vertices[:, 1],
                                        z=gtmesh.vertices[:, 2],
                                        i=I, j=J, k=K, name='gt',
                                        color=color, opacity=1.0, flatshading=False,
                                        lighting=dict(diffuse=1, ambient=0, specular=0),
                                        lightposition=dict(x=0,y=0,z=-1))
                        if (with_opt):
                            fig.add_trace(trace, row=1, col=2)
                        else:
                            fig.add_trace(trace, row=1, col=3)


                    div = offline.plot(fig, include_plotlyjs=False, output_type='div', auto_open=False)
                    div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
                    if (with_opt):
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
                    else:
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
                                            }}
            
                                        isUnderRelayout = true;
                                        }})
                                        </script>'''.format(div_id=div_id)
                    # merge everything
                    div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
#                    logging.info (ds.shapenames[data[-1]])
                    with open(os.path.join(path, "compare_{0}.html".format(ds.npyfiles_mnfld[data[-1][0].item()].split('/')[-3] + '_' + ds.npyfiles_mnfld[data[-1][0].item()].split('/')[-1].split('.npy')[0])),
                            "w") as text_file:
                        text_file.write(div)
        i = i + 1
        logging.info (i)
    if (index == -1):
        pd.DataFrame(chamfer_results).to_csv(os.path.join(path,"eval_results.csv"))
        # with open(os.path.join(path,"chamfer.csv"),"w",) as f:
        #     if (with_opt):
        #         f.write("shape, chamfer_dist, chamfer scan dist, after opt chamfer dist, after opt chamfer scan dist\n")
        #         for result in chamfer_results:
        #             f.write("{}, {} , {}\n".format(result[0], result[1], result[2], result[3], result[4]))
        #     else:
        #         f.write("shape, chamfer_dist, chamfer scan dist\n")
        #         for result in chamfer_results:
        #             f.write("{}, {} , {}\n".format(result[0], result[1], result[2]))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--expname", required=False, help='The experiment name to be evaluated.',default='')
    arg_parser.add_argument("--override", required=False, help='Override exp name.',default='')
    arg_parser.add_argument("--exps_folder_name", default="exps", help='The experiments directory.')
    arg_parser.add_argument("--timestamp", required=False, default='latest')
    arg_parser.add_argument("--conf", required=False , default='./confs/dfaust_local.conf')
    arg_parser.add_argument("--checkpoint", help="The checkpoint to test.", default='latest')
    arg_parser.add_argument("--split", required=False,help="The split to evaluate.",default='')
    arg_parser.add_argument("--parallel", default=False, action="store_true", help="Should be set to True if the loaded model was trained in parallel mode")
    arg_parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto].')
    arg_parser.add_argument('--with_opt', default=False, action="store_true", help='If set, optimizing latent with reconstruction Loss versus input scan')
    arg_parser.add_argument('--resolution', default=256, type=int, help='Grid resolution')
    arg_parser.add_argument('--index', default=-1, type=int, help='')
    arg_parser.add_argument('--chamfer_only', default=False,action="store_true")
    arg_parser.add_argument('--recon_only', default=False,action="store_true")
    arg_parser.add_argument('--plot_cmpr', default=False,action="store_true")
    arg_parser.add_argument('--is_action', default=False,action="store_true")


    
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logging.info ("running")

    args = arg_parser.parse_args()
    
    if args.gpu != 'ignore':
        if args.gpu == "auto":
            deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                        excludeUUID=[])
            gpu = deviceIDs[0]
        else:
            gpu = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    evaluate_with_load(gpu=args.gpu,
                       parallel=args.parallel,
                       conf=args.conf,
                       exps_folder_name=args.exps_folder_name,
                       timestamp=args.timestamp,
                       checkpoint=args.checkpoint,
                       resolution=args.resolution,
                       override=args.override,
                       eval_train=False,
                       is_action=args.is_action)

