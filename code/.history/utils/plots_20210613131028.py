import plotly.graph_objs as go
import plotly.offline as offline
import torch
import numpy as np
from skimage import measure
import os
from tqdm import tqdm
import utils.general as utils

def get_scatter_trace(points,name,caption = None,colorscale = None,color = None):

    # assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    # assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "
    if (type(points) == list):
        if points[0][0].shape[-1] == 3:
            trace = [go.Scatter3d(
                x=p[0][:, 0],
                y=p[0][:, 1],
                z=p[0][:, 2],
                mode='markers',
                name=p[1],
                marker=dict(
                    size=3,
                    line=dict(
                        width=2,
                    ),
                    opacity=0.9,
                    colorscale=colorscale,
                    showscale=True,
                    color=p[2],
                ), text=p[2] if len(p) == 3 else None) for p in points]
        else:
            trace = [go.Scatter(
                x=p[:,0],
                y=p[:,1],
                mode='markers',
                marker=dict(
                    size=3,
                    line=dict(
                        width=1,
                    ),
                    opacity=0.9,
                    colorscale=colorscale,
                    showscale=True,
                    color=color,
                ), text=caption) for p in points]

    else:
        points = points.detach()
        if points.shape[-1] == 3:
            trace = [go.Scatter3d(
                x=points[:,0],
                y=points[:,1],
                z=points[:,2],
                mode='markers',
                name=name,
                marker=dict(
                    size=3,
                    line=dict(
                        width=2,
                    ),
                    opacity=0.9,
                    colorscale=colorscale,
                    showscale=False,
                    color=color,
                ), text=caption)]
        else:
            trace = [go.Scatter(
                x=points[:,0],
                y=points[:,1],
                mode='markers',
                marker=dict(
                    size=3,
                    line=dict(
                        width=1,
                    ),
                    opacity=1.0,
                    colorscale=colorscale,
                    showscale=True,
                    color=color,
                ), text=caption)]

    return trace

def plot_threed_scatter(points,path,epoch,in_epoch):
    trace = get_scatter_trace(points,'pnts')
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                           yaxis=dict(range=[-2, 2], autorange=False),
                                                           zaxis=dict(range=[-2, 2], autorange=False),
                                                           aspectratio=dict(x=1, y=1, z=1)))

    fig1 = go.Figure(data=trace, layout=layout)

    filename = '{0}/scatter_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)
    offline.plot(fig1, filename=filename, auto_open=False)

def plot_surface(with_points, points, decoder, latent, path, epoch, in_epoch,
                 shapefile, resolution, mc_value, is_uniform_grid, verbose, save_html, save_ply, overwrite, is_3d,z_func):
    if (is_uniform_grid):
        filename = '{0}/uniform_iteration_{1}_{2}'.format(path, epoch, in_epoch)
    else:
        filename = '{0}/nonuniform_iteration_{1}_{2}'.format(path, epoch, in_epoch)

    if (not os.path.exists(filename) or overwrite):
        if (with_points):
            res = decoder(points.unsqueeze(0), None,None, latent=latent, only_decoder_forward=True, only_encoder_forward=False)
            pnts_val = res
            pnts_val = pnts_val.cpu()
            points = points.cpu()
            caption = ["decoder : {0}".format(val.item()) for val in pnts_val.squeeze()]

        for key,val in z_func.items():
            surface = get_surface_trace(points,decoder,  latent, resolution, mc_value, is_uniform_grid ,verbose, save_ply,name='reconstruction', is_3d=is_3d,z_func=val)
            trace_surface = surface["mesh_trace"]
            layout = go.Layout(title= go.layout.Title(text="epoch : {0} <br> input filename:{1}".format(epoch, shapefile)), width=1200, height=1200, scene=dict(camera=dict(up=dict(x=0, y=1, z=0),center=dict(x=0, y=0.0, z=0),eye=dict(x=0, y=0.6, z=0.9)),xaxis=dict(range=[-2, 2], autorange=False),
                                                                                                            yaxis=dict(range=[-2, 2], autorange=False),
                                                                                                            zaxis=dict(range=[-2, 2], autorange=False),
                                                                                                            aspectratio=dict(x=1, y=1, z=1)))
            if (with_points):
                trace_pnts = get_scatter_trace(points[:, -3:],name="input", caption=caption)
                
                fig1 = go.Figure(data=trace_pnts + trace_surface, layout=layout)
            else:
                fig1 = go.Figure(data=trace_surface, layout=layout)


            if (save_html  or not is_3d ):
                offline.plot(fig1, filename=filename + '_' + key + '.html', auto_open=False)
            if (not surface['mesh_export'] is None):
                surface['mesh_export'].export(filename + '_' + key +  '.ply', 'ply')
        return surface['mesh_export'], fig1

def get_surface_trace(points, decoder, latent, resolution, mc_value, is_uniform, verbose, save_ply, grid_boundary=2.0,is_3d=True,name='',z_func=None):

    trace = []
    meshexport = None

    if (is_uniform):
        grid = get_grid_uniform(None,resolution,grid_boundary,is_3d)
    else:
        grid = get_grid(points[:,-3:],resolution)

    z = []

    for pnts in tqdm(torch.split(grid['grid_points'], 10000, dim=0)):

        # if (not latent is None):
        #     pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        #print ('before : {0}'.format(pnts.shape))
        if hasattr(decoder,"device_ids") and len(decoder.device_ids) > 1:
            v = decoder(pnts, pnts,sample_nonmnfld=None, latent=latent.repeat(len(decoder.device_ids),1), only_encoder_forward=False, only_decoder_forward=True)
        else:
            v = decoder(pnts, pnts,sample_nonmnfld=None, latent=latent, only_encoder_forward=False, only_decoder_forward=True)
        if type(v) is tuple:
            v = v[0]
        #print ('after : {0}'.format(v.shape))
        v = v.squeeze().detach().cpu().numpy()
        if not z_func is None:
            v = z_func(v)
        z.append(v)
    z = np.concatenate(z,axis=0)

    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        import trimesh
        z  = z.astype(np.float64)

        if is_3d :
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=mc_value,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1]))
        
            

            verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
            if (save_ply):
                meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)


            def tri_indices(simplices):
                return ([triplet[c] for triplet in simplices] for c in range(3))

            I, J, K = tri_indices(faces)

            color = '#ffffff'
            trace.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, 
                                color=color, opacity=1.0, flatshading=False,
                                lighting=dict(diffuse=1, ambient=0, specular=0),
                                lightposition=dict(x=0, y=0, z=-1),
                                showlegend=True,name=name))
        else:
            trace.append(go.Contour(x=grid['xyz'][0],y=grid['xyz'][1],z=z.reshape(resolution,resolution),
            line=dict(width=6),
            hoverinfo='skip',
            autocontour=False,contours=dict(
            start=0,
            end=0,
            size=0,
            coloring="none"
        )))
            meshexport = None
            
    return {"mesh_trace":trace,
            "mesh_export":meshexport}

def plot_cuts_axis(points,decoder,latent,path,epoch,near_zero,axis,file_name_sep='/'):
    onedim_cut = np.linspace(-1.0, 1.0, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_axis = points[:,axis].min(dim=0)[0].item()
    max_axis = points[:,axis].max(dim=0)[0].item()
    mask = np.zeros(3)
    mask[axis] = 1.0
    if (axis == 0):
        position_cut = np.vstack(([np.zeros(xx.shape[0]), xx, yy]))
    elif (axis == 1):
        position_cut = np.vstack(([xx,np.zeros(xx.shape[0]), yy]))
    elif (axis == 2):
        position_cut = np.vstack(([xx, yy, np.zeros(xx.shape[0])]))
    position_cut = [position_cut + i*mask.reshape(-1, 1) for i in np.linspace(min_axis - 0.1, max_axis + 0.1, 2)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = utils.get_cuda_ifavailable(torch.tensor(pos.T, dtype=torch.float))
        z = []
        for i, pnts in enumerate(torch.split(field_input, 10000, dim=0)):
            if (not latent is None):
                pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
            v = decoder(pnts)
            if (type(v) is tuple):
                v = v[0]
            z.append(v.detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            if (np.min(z) < -1.0e-5):
                start = -0.1
            else:
                start = 0.0
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=start,
                                     end=0.1,
                                     size=0.01
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                ncontours=70
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='axis {0} = {1}'.format(axis,pos[axis, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}{1}cutsaxis_{2}_{3}_{4}.html'.format(path,file_name_sep,axis, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)

def plot_cuts(points,decoder,path,epoch,in_epoch,near_zero,latent,number_of_cuts=2,z_func=None):
    onedim_cut = np.linspace(-2, 2, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, number_of_cuts)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = utils.get_cuda_ifavailable(torch.tensor(pos.T, dtype=torch.float))
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            # if (not latent is None):
            #     input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            v = decoder(input_.unsqueeze(0),latent)
            if type(v) is tuple:
                v = v[0]
            
            v = v.detach().cpu().numpy()
            if not z_func is None:
                v = z_func(v)
            z.append(v)
        z = np.concatenate(z, axis=0)

        if (near_zero):
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.001,
                                     end=0.001,
                                     size=0.00001
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='y = {0}'.format(pos[1, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}/cuts{1}_{2}_{3}.html'.format(path, epoch,in_epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)


def get_grid(points,resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().detach().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().detach().cpu().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis}

def get_grid_uniform(points,resolution,grid_boundary,is_3d):
    if is_3d:
        x = np.linspace(-1,1, resolution)
        length = np.max(x) - np.min(x)
        y = np.linspace(-1,2, resolution)
        y = np.arange(-1, 2 + length / (x.shape[0] - 1), length / (x.shape[0] - 1) )
        z = np.linspace(-1,1, resolution)

        shortest_axis_length = x.max().item() - x.min()
        length = np.max(x) - np.min(x)    
        shortest_axis_length = x.max().item() - x.min()
    
        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = utils.get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float))
    else:
        x = np.linspace(-2.0,2.0, resolution)
        shortest_axis_length = x.max().item() - x.min()
        y = x
        z = x

        xx, yy = np.meshgrid(x, y)
        grid_points = utils.get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel()]).T, dtype=torch.float))
        

    return {"grid_points": grid_points,
            "shortest_axis_length": shortest_axis_length,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}