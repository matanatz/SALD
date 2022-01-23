from  model.network_linear_prob import DeformNetwork
from pyhocon import ConfigFactory
import torch
from torch import nn
import numpy as np
import utils.general as utils
import GPUtil
import os
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.subplots import make_subplots
import utils.plots as plt
from  datasets.datasets_syn import ElipseRotDataSet
from evaluate.eval_3d import evaluate
g_i  = 1
class CubeFlow(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.arr = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
               np.array([0, 0, 1]), np.array([0, 0, -1])]
        self.arr = torch.tensor(np.stack(self.arr)).float()
        self.e3 = utils.get_cuda_ifavailable(torch.eye(3))
        self.e9 = utils.get_cuda_ifavailable(torch.eye(9))
        self.e3.requires_grad_(True)
        self.e9.requires_grad_(True)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, input, latent, compute_grad=False, cat_latent=True, grad_with_repsect_to_latent=False):
    #def forward(self,con,compute_grad=True):
        
        if len(input.shape) == 2:
            con = torch.cat([input,latent.repeat(input.shape[0],1)  * 1.5],dim=-1)
            con = con.unsqueeze(0)
        else:    
            con = torch.cat([input,latent.unsqueeze(1).repeat(1,input.shape[1],1)  *10],dim=-1)   
        con = con.requires_grad_(True)
        u = (torch.cos(con[:,:,3:]) * self.e3[0:1, :].unsqueeze(1) + torch.sin(con[:,:,3:]) * self.e3[1:2, :].unsqueeze(1) +  -con[:,:,3:] * self.e3[2:3, :].unsqueeze(1))/torch.sqrt(1 + con[:,:,3:]**2)
        u = torch.cos(0*con[:,:,3:]) * self.e3[2:3, :].unsqueeze(1)
        #print (torch.norm(u,p=2,dim=-1))

        Rt = (torch.cos(con[:,:,3:]) + u[..., 0:1] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[0:1, :].unsqueeze(1) + \
             (u[..., 1:2] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) - u[..., 2:3] * torch.sin(con[:,:,3:])) * self.e9[1:2, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) + u[..., 1:2] * torch.sin(con[:,:,3:])) * self.e9[2:3, :].unsqueeze(1) + \
             (u[..., 1:2] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) + u[..., 2:3] * (torch.sin(con[:,:,3:]))) * self.e9[3:4, :].unsqueeze(1) + \
             (torch.cos(con[:,:,3:]) + u[..., 1:2] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[4:5, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 1:2] * (1 - torch.cos(con[:,:,3:])) - u[..., 0:1] * torch.sin(con[:,:,3:])) * self.e9[5:6, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) - u[..., 1:2] * (torch.sin(con[:,:,3:]))) * self.e9[6:7, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 1:2] * (1 - torch.cos(con[:,:,3:])) + u[..., 0:1] * torch.sin(con[:,:,3:])) * self.e9[7:8, :].unsqueeze(1) + \
             (torch.cos(con[:,:,3:]) + u[..., 2:3] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[8:9, :].unsqueeze(1)

        R = Rt.view(-1,con.shape[1] , 3, 3)

        C = con[...,3:] * torch.tensor([1,0,0]).to(R).float().unsqueeze(0).unsqueeze(0)

        A = torch.tensor([[1.4, 0.0, 0.0],
                          [0.0, 1.4, 0.0],
                          [0.0, 0.0, 0.2]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R.transpose(2, 3), con[..., :3] - C)
        s = 1
        #f1 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5
        f1 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', con[..., :3] -s*C, A), con[..., :3] -  s*C) - 0.5

        A = torch.tensor([[3.8, 0.0, 0],
                          [0.0, 0.6, 0.0],
                          [0, 0.0, 3.8]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R.transpose(2,3), con[..., :3] - s*C + torch.tensor([0, 0, 0.4]).unsqueeze(0).to(con))
        f2 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5

        A = torch.tensor([[0.35, 0.0, 0],
                          [0.0, 2.8, 0],
                          [0, 0, 2.8]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R, con[..., :3] - s*C - torch.tensor([0, 0, 0.6]).unsqueeze(0).to(con))
        f3 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5

        eval = torch.min(f1, torch.min(f2, f3)).squeeze()

        probs = self.fc2(self.relu(self.fc1(con)))
        #probs = torch.cat([input[...,2:3],-100*input[...,2:3]],dim=-1)
        if compute_grad or grad_with_repsect_to_latent:
            grad = torch.autograd.grad(eval,
                                       con,
                                       torch.ones_like(eval),
                                       retain_graph=True,
                                       create_graph=True)[0]
            return eval,grad,con,probs
        else:
            return  eval,None,con,probs
        


class Cube_2Flow(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.arr = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
               np.array([0, 0, 1]), np.array([0, 0, -1])]
        self.arr = torch.tensor(np.stack(self.arr)).float()
        self.e3 = utils.get_cuda_ifavailable(torch.eye(3))
        self.e9 = utils.get_cuda_ifavailable(torch.eye(9))
        self.e3.requires_grad_(True)
        self.e9.requires_grad_(True)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        number_of_shapes = 100
        rotations = np.random.randn(number_of_shapes,3,3)
        self.rotations = torch.cat([torch.eye(3).unsqueeze(0), torch.tensor(np.concatenate([np.expand_dims(np.linalg.qr(x)[0],0) for x in rotations])).float()],dim=0)
        self.centers = torch.cat([torch.zeros([1,3]),2 * torch.tensor(np.random.rand(number_of_shapes,3))],dim=0).float()
        scales = []
        for j in range(number_of_shapes):
            a = 1#(1.4 - 0.6)*np.random.rand() + 0.6
            b = 1#(1.4 - 0.6) * np.random.rand() + 0.6
            c = 1#(1.4 - 0.6) * np.random.rand() + 0.6
            scales.append(torch.tensor([[1./a,0.0,0.0],[0.0,1./b,0.0],[0.0,0.0,1./c]]).float().unsqueeze(0))
            #scales.append(torch.tensor([[1.,0.0,0.0],[0.0,1.,0.0],[0.0,0.0,1.]]).float().unsqueeze(0))
        scales = torch.cat(scales,dim=0)
        self.scales = torch.cat([torch.eye(3).unsqueeze(0), scales],dim=0)
        

    def forward(self, input, latent, compute_grad=False, cat_latent=True, grad_with_repsect_to_latent=False):
    #def forward(self,con,compute_grad=True):
        global g_i
        if len(input.shape) == 2:
            con = torch.cat([input,latent.repeat(input.shape[0],1)  * 1.5],dim=-1)
            con = con.unsqueeze(0)
        else:    
            con = torch.cat([input,latent.unsqueeze(1).repeat(1,input.shape[1],1)  *10],dim=-1)   
        con = con.requires_grad_(True)
        u = (torch.cos(con[:,:,3:]) * self.e3[0:1, :].unsqueeze(1) + torch.sin(con[:,:,3:]) * self.e3[1:2, :].unsqueeze(1) +  -con[:,:,3:] * self.e3[2:3, :].unsqueeze(1))/torch.sqrt(1 + con[:,:,3:]**2)
        u = torch.cos(0*con[:,:,3:]) * self.e3[2:3, :].unsqueeze(1)
        #print (torch.norm(u,p=2,dim=-1))

        Rt = (torch.cos(con[:,:,3:]) + u[..., 0:1] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[0:1, :].unsqueeze(1) + \
             (u[..., 1:2] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) - u[..., 2:3] * torch.sin(con[:,:,3:])) * self.e9[1:2, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) + u[..., 1:2] * torch.sin(con[:,:,3:])) * self.e9[2:3, :].unsqueeze(1) + \
             (u[..., 1:2] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) + u[..., 2:3] * (torch.sin(con[:,:,3:]))) * self.e9[3:4, :].unsqueeze(1) + \
             (torch.cos(con[:,:,3:]) + u[..., 1:2] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[4:5, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 1:2] * (1 - torch.cos(con[:,:,3:])) - u[..., 0:1] * torch.sin(con[:,:,3:])) * self.e9[5:6, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 0:1] * (1 - torch.cos(con[:,:,3:])) - u[..., 1:2] * (torch.sin(con[:,:,3:]))) * self.e9[6:7, :].unsqueeze(1) + \
             (u[..., 2:3] * u[..., 1:2] * (1 - torch.cos(con[:,:,3:])) + u[..., 0:1] * torch.sin(con[:,:,3:])) * self.e9[7:8, :].unsqueeze(1) + \
             (torch.cos(con[:,:,3:]) + u[..., 2:3] ** 2 * (1 - torch.cos(con[:,:,3:]))) * self.e9[8:9, :].unsqueeze(1)

        R = Rt.view(-1,con.shape[1] , 3, 3)

        C = con[...,3:] * torch.tensor([0.1,-0.1,0.3]).to(R).float().unsqueeze(0).unsqueeze(0)

        return torch.abs(torch.bmm(self.rotations[g_i].unsqueeze(0).transpose(1,2).repeat(con.shape[0],1,1).to(con),(con[...,:3] - self.centers[g_i].to(con)).transpose(1,2)).transpose(1,2)).max(-1)[0] - 0.5

        

        A = torch.tensor([[1.4, 0.0, 0.0],
                          [0.0, 1.4, 0.0],
                          [0.0, 0.0, 0.2]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R.transpose(2, 3), con[..., :3] - C)
        s = 1
        #f1 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5
        f1 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', con[..., :3] -s*C, A), con[..., :3] -  s*C) - 0.5

        A = torch.tensor([[3.8, 0.0, 0],
                          [0.0, 0.6, 0.0],
                          [0, 0.0, 3.8]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R.transpose(2,3), con[..., :3] - s*C + torch.tensor([0, 0, 0.4]).unsqueeze(0).to(con))
        f2 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5

        A = torch.tensor([[0.35, 0.0, 0],
                          [0.0, 2.8, 0],
                          [0, 0, 2.8]]).float().to(con)
        cc = torch.einsum('bpij,bpj->bpi', R, con[..., :3] - s*C - torch.tensor([0, 0, 0.6]).unsqueeze(0).to(con))
        f3 = torch.einsum('bpj,bpj->bp', torch.einsum('bpk,kj->bpj', cc, A), cc) - 0.5

        eval = torch.min(f1, torch.min(f2, f3)).squeeze()

        probs = self.fc2(self.relu(self.fc1(con)))
        #probs = torch.cat([input[...,2:3],-100*input[...,2:3]],dim=-1)
        if compute_grad or grad_with_repsect_to_latent:
            grad = torch.autograd.grad(eval,
                                       con,
                                       torch.ones_like(eval),
                                       retain_graph=True,
                                       create_graph=True)[0]
            return eval,grad,con,probs
        else:
            return  eval,None,con,probs
        


conf_all = """
train.base_path = .
train.auto_decoder = True
train.sigma = 0.0
"""

conf_all = ConfigFactory.parse_string(conf_all)

conf = """
{
    t_samples = 1
    predict_normals_on_surfce = False
    viscosity = 0.0
    with_vae = False
    with_sample = True
    uniform_sample=True
    sample_box = [1.0,-1.0,2.0,-1.0,1.0,-1.0]
    rand_sample_factor = 1
    v_sample_factor = 4
    v_projection_steps = 1
    v_start_uniform = True
    proj_with_con = False
    v_filter = 0.1
    v_noise = 0.02
    con_dir = False
    dir_detach=True
    dist_sample_factor = 2
    concate_proj = False
    t_include_bndry = False
    t_beta_sampling = False
    is_nearest = False
    dist_start_uniform = True
    is_mean_shape = False
    detach_f = False
    lambda_i = 0.01
    adapted_lr = True
    K = 2
    weighted_lsq = False
    sigma_square = 0
    noise_lsq = 0

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
}
"""
deviceIDs = GPUtil.getAvailable(
                order="memory",
                limit=1,
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
gpu = deviceIDs[0]
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
t = torch.tensor([[0.0],[1.0]]).float().cuda()
ds = ElipseRotDataSet(split=None,number_of_points=24,number_of_shapes=2,rots=t)
conf = ConfigFactory.parse_string(conf)
net = DeformNetwork(conf,1,True).cuda()
cube_flow = Cube_2Flow().cuda() 
net.implicit_map = cube_flow


utils.mkdir_ifnotexists('./test')
utils.mkdir_ifnotexists('./test/ellipse')
utils.mkdir_ifnotexists('./test/ellipse/1')


fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]],
                                            subplot_titles=[""])


for i in range(20):
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scene"}]],
                                            subplot_titles=[""])
    fig.layout.scene.update(dict(camera=dict(center= {'x': 0, 'y': 0, 'z': 0},
                                         eye={'x': 0.7953342841249091,'y': 0.7953342841249094,'z': 0.7953342841249093},
                                         up={'x': -0.0021232823577897373, 'y': 0.0021278002990373383, 'z': 0.9999954820587524}),xaxis=dict(range=[-3, 3], autorange=False,showbackground=False,visible=False),
                                                yaxis=dict(range=[-3, 3],showbackground=False, visible=False,autorange=False),
                                                zaxis=dict(range=[-3, 3],showbackground=False, visible=False,autorange=False),
                                                aspectratio=dict(x=1, y=1, z=1)))
    g_i = i
    reconstruction = plt.plot_surface(with_points=False,
                                            points=None,
                                            decoder=net,
                                            latent=t[0],
                                            path='./test/ellipse',
                                            epoch='a',
                                            in_epoch='b',
                                            shapefile='t',
                                            resolution=256,
                                            mc_value=0,
                                            is_uniform_grid=True,
                                            verbose=True,
                                            save_html=False,
                                            save_ply=True,
                                            overwrite=True,
                                            is_3d=True,
                                            z_func={'id':lambda x:x})

    def tri_indices(simplices):
                            return ([triplet[c] for triplet in simplices] for c in range(3))


    I, J, K = tri_indices(reconstruction.faces)
    color = '#ffffff'
    trace = go.Mesh3d(x=reconstruction.vertices[:, 0], y=reconstruction.vertices[:, 1],
                    z=reconstruction.vertices[:, 2],
                    i=I, j=J, k=K, name='scan',
                    color=color, opacity=1.0, flatshading=False,
                    lighting=dict(diffuse=1, ambient=0, specular=0), lightposition=dict(x=0, y=0, z=-1))
    fig.add_trace(trace, row=1, col=1)

    fig.write_image('test/{0}.png'.format(i),width=1200,height=1200)
    div = offline.plot(fig, include_plotlyjs=False, output_type='div', auto_open=False)
    div_id = div.split('=')[1].split()[0].replace("'", "").replace('"', '')
    js = '''
                                            <script>
                                            var gd = document.getElementById('{div_id}');
                                            var isUnderRelayout = false
                                            
                                            gd.on('plotly_relayout', () => {{
                                            
                                            console.log(gd.layout.scene.camera);
                                            
                
                                            isUnderRelayout = true;
                                            }})
                                            </script>'''.format(div_id=div_id)
                        # merge everything
    div = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' + div + js
    #                    logging.info (ds.shapenames[data[-1]])
    with open(os.path.join('./test/ellipse', "compare.html"),"w") as text_file:
        text_file.write(div)


# evaluate(network=net,
#                                 exps_folder_name='./test',
#                                 experiment_name='ellipse',
#                                 timestamp='1',
#                                 ds=ds,
#                                 epoch=0,
#                                 with_opt=False,
#                                 resolution=256,
#                                 with_normals=False,
#                                 conf=conf_all,
#                                 index=-1,
#                                 chamfer_only=False,
#                                 recon_only=False,
#                                 lat_vecs=t,
#                                 lat_vecs_sigma=None,
#                                 with_cuts=False,
#                                 visdomer=None,
#                                 env='',
#                                 step_log=None,
#                                 with_video=False,
#                                 reuse_window=True,
#                                 with_optimize_line=False,
#                                 all_couples=2,
#                                 no_k=True,length=60,cancel_ffmpeg=False,save_animation=False)