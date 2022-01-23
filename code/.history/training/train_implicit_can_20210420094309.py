import numpy as np
import os
import torch
import sys
import logging
import time
import plotly.offline as offline
import plotly.graph_objs as go
import utils.general as utils
import evaluate.eval_dfaust as eval
import evaluate.eval_3d as eval_3d
import evaluate.eval_2d as eval_2d
from utils.plots import plot_cuts,get_scatter_trace,plot_surface,plot_threed_scatter
from training.train import BaseTrainRunner
from plotly.subplots import make_subplots
from itertools import chain
import pandas as pd
import utils.plots as plt

class TrainImplicitCanRunner(BaseTrainRunner):
    def register_to_monitor(self):
        self.step_log.add('total_loss')
        self.step_log.add('reconstruction_loss')
        self.step_log.add('canonical_eikonal_loss')
    
    def run(self):
        win = None
        timing_log = []
        loss_log_epoch = []
        lr_log_epoch = []
        logging.debug("*******************running*********")
        self.epoch = 0
        for epoch in range(self.start_epoch, self.nepochs + 2):
            self.epoch = epoch
            if (epoch in self.test_epochs and self.ds_len > 1):
                
                self.network.eval()
                if self.conf.get_bool('plot.is_3d'):
                    if self.conf.get_bool('train.plot_corr'): 
                        eval_3d.evaluate(network=self.network,
                                exps_folder_name=self.exps_folder_name,
                                experiment_name=self.expname,
                                timestamp=self.timestamp,
                                ds=self.ds,
                                epoch=epoch,
                                with_opt=self.conf.get_bool('train.auto_decoder'),
                                resolution=32,
                                with_normals=self.conf.get_bool('network.encoder.with_normals'),
                                conf=self.conf,
                                index=-1,
                                chamfer_only=False,
                                recon_only=False,
                                lat_vecs=self.lat_vecs,
                                lat_vecs_sigma=None,
                                with_cuts=epoch%5000 == 0 and False,
                                visdomer=self.visdomer,
                                env=self.visdom_env,
                                step_log=pd.DataFrame(self.step_log),
                                with_video=False,
                                reuse_window=True,
                                with_optimize_line=False,
                                no_k=True,
                                cancel_ffmpeg=True,
                                save_animation=True,
                                all_couples=self.conf.get_int('train.plot_num_couples') if epoch % 10000 != 0 else 10)
                    else:
                        eval.evaluate(network=self.network,
                            exps_folder_name=self.exps_folder_name,
                            experiment_name=self.expname,
                            timestamp=self.timestamp,
                            epoch=epoch,
                            resolution=128,
                            conf=self.conf,
                            index=-1,
                            chamfer_only=False,
                            recon_only=False,
                            lat_vecs=self.lat_vecs)
                    # pass
                else:
                    eval_2d.evaluate(network=self.network,
                            exps_folder_name=self.exps_folder_name,
                            experiment_name=self.expname,
                            timestamp=self.timestamp,
                            split_filename=self.test_split_file,
                            epoch=epoch,
                            with_opt=self.conf.get_bool('train.auto_decoder'),
                            resolution=64,
                            with_normals=self.conf.get_bool('network.encoder.with_normals'),
                            conf=self.conf,
                            index=-1,
                            chamfer_only=False,
                            recon_only=False,
                            lat_vecs=self.lat_vecs)
                
                #self.optimizer.zero_grad()
                self.network.train()


            start_epoch = time.time()
            batch_loss = 0.0

            if (epoch % self.conf.get_int('train.save_checkpoint_frequency') == 0 or epoch == self.start_epoch) and epoch > 0:
                self.save_checkpoints()
            if epoch % self.conf.get_int('train.plot_frequency') == 0 and epoch > 0:
                logging.debug("in plot")
                self.network.eval()
                for i in range(min(4, self.ds_len)):
                    pnts, normals, sample, idx = next(iter(self.eval_dataloader))
                    pnts = pnts.cuda()
                    normals = normals.cuda()
                    idx = idx.cuda()


                    if self.latent_size > 0:
                        if self.conf.get_bool('train.auto_decoder'):
                            latent = self.lat_vecs(idx)
                        else:
                            _,latent,_ = self.network(manifold_points=pnts, manifold_normals=normals,sample_nonmnfld=None, latent=None, latent_sigma_inputs = None, only_encoder_forward=True, only_decoder_forward=False)

                        pnts = pnts[0]
                    else:
                        latent = None
                        pnts = pnts[0]

                    logging.debug("before plot")
                    plot_surface(with_points=True,
                                points=pnts,
                                decoder=self.network,
                                latent=latent,
                                path=self.plots_dir,
                                epoch=epoch,
                                in_epoch=i,
                                shapefile=self.ds.npyfiles_mnfld[idx],
                                z_func={'id':lambda x:x},#, 'log':lambda x : (-1.0/10.0) * np.log(x + 1)},
                                **self.conf.get_config('plot'))
                    logging.debug("after plot")                

            self.network.train()
            if (self.adjust_lr):
                self.adjust_learning_rate(epoch)
            logging.debug('before data loop {0}'.format(time.time()-start_epoch))
            before_data_loop = time.time()
            data_index = 0
            for pnts_mnfld,normals_mnfld,sample_nonmnfld,indices in self.dataloader:
                logging.debug('in loop data {0}'.format(time.time()-before_data_loop))
                start = time.time()
                pnts_mnfld = pnts_mnfld.cuda()
                normals_mnfld = normals_mnfld.cuda()
                sample_nonmnfld = sample_nonmnfld.cuda()
                indices = indices.cuda()
                #logging.debug('after cuda {0}'.format(time.time()-start))

                if self.conf.get_bool('train.auto_decoder') and self.latent_size > 0:
                    latent_inputs = []
                    latent_inputs = self.lat_vecs(indices)
                    latent_sigma_inputs = None
                    #latent_sigma_inputs = self.lat_vecs_sigma(indices) if not self.lat_vecs_sigma is None else None

                else:
                    latent_inputs = None
                    latent_sigma_inputs = None

                outputs = self.network(pnts_mnfld, None, sample_nonmnfld[:,:,:3], latent_inputs, False, False,epoch=epoch)

                if data_index == 0 and self.conf.get_bool('train.debug_proj') and epoch%1000 == 0 and epoch >= 0 and 'debug_v_projection_end' in outputs:
                    for id_rand in range(min(outputs['debug_v_projection_end'].shape[0],1)):
                    #id_rand = np.random.randint(outputs['debug_v_projection_end'].shape[0])
                        pnts_trace = plt.get_scatter_trace([(outputs['debug_v_projection_start'][id_rand].detach().cpu(),'start',None),
                                                            (outputs['debug_v_projection_end'][id_rand].detach().cpu(),'end',outputs['debug_v_projection_end_eval'][id_rand].squeeze().cpu().detach().numpy()),
                                                            (outputs['debug_v_projection_end'][id_rand].detach().cpu(),'filter',100*outputs['v_filter'][id_rand].squeeze().cpu().detach().numpy()),
                                                            #(outputs['debug_v_projection_end'][id_rand].detach().cpu(),'dv',(outputs['dv'][id_rand]**2).sum([-1,-2]).squeeze().cpu().detach().numpy())
                                                            ] + ([(outputs['debug_v_projection_end'][id_rand].detach().cpu(),'cond',outputs['debug_cond'][id_rand].detach().cpu()),
                                                                  (outputs['debug_v_projection_end'][id_rand].detach().cpu(),'cond_above-4',outputs['debug_cond_cnt'][id_rand].detach().cpu())] if 'debug_cond' in outputs  else []))
                        surface = plt.get_surface_trace(None,self.network,  outputs['debug_latent'][id_rand:id_rand+1], 64, 0.0, True ,True, False, is_3d=True,z_func=None)
                        layout = go.Layout( width=1200, height=1200, scene=dict(xaxis=dict(range=[-3, 3], autorange=False),
                                                                    yaxis=dict(range=[-3, 3], autorange=False),
                                                                    zaxis=dict(range=[-3, 3], autorange=False),
                                                                    aspectratio=dict(x=1, y=1, z=1)))
                        fig1 = go.Figure(data=pnts_trace + surface['mesh_trace'], layout=layout)
                        offline.plot(fig1, filename=self.plots_dir + '/_debug{0}_{1}.html'.format(epoch,id_rand), auto_open=False)

                    if 'debug_density' in outputs:
                        traces = []
                        traces = [go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density01'].cpu().detach().numpy(),mode='lines',name='0_to_1',visible='legendonly'),
                                  go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density10'].cpu().detach().numpy(),mode='lines',name='1_to_0',visible='legendonly'),
                                  go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density02'].cpu().detach().numpy(),mode='lines',name='0_to_2',visible='legendonly'),
                                  go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density20'].cpu().detach().numpy(),mode='lines',name='2_to_0',visible='legendonly'),
                                  go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density12'].cpu().detach().numpy(),mode='lines',name='1_to_2',visible='legendonly'),
                                  go.Scatter(x=outputs['debug_density']['t'].cpu().detach().numpy(),y=outputs['debug_density']['density21'].cpu().detach().numpy(),mode='lines',name='2_to_1',visible='legendonly')]
                        fig1 = go.Figure(data=traces, layout=layout)                                                                    
                        offline.plot(fig1, filename=self.plots_dir + '/_debug_density{0}.html'.format(epoch), auto_open=False)

                #logging.debug('after forward {0}'.format(time.time()-start))
                
                 
                loss_res = self.loss(network_outputs=outputs, normals_gt=normals_mnfld, normals_nonmnfld_gt = sample_nonmnfld[:,:,3:6], pnts_mnfld=pnts_mnfld, gt_nonmnfld=sample_nonmnfld[:,:,-1],epoch=epoch)
                #logging.debug('after loss forward {0}'.format(time.time()-start))

                loss = loss_res["loss"].mean()

                #logging.debug('before backward  {0}'.format(time.time()-start))
                start_back = time.time()
                loss.backward()
                logging.debug('after backward delta  {0}'.format(time.time()-start_back))
                # for param in self.network.parameters():
                #     if (param.grad.isnan().any()):
                #         print ("nan")
                logging.debug('after backward  {0}'.format(time.time()-start))
                self.optimizer.step()

                if 'total_loss' in self.step_log:
                    len_step_loss = len(self.step_log['total_loss'])
                else:
                    len_step_loss = 0
                for k in loss_res['loss_monitor'].keys():
                    if k in self.step_log:
                        self.step_log[k].append(loss_res['loss_monitor'][k].mean().item())
                    else:
                        if len_step_loss > 0:
                            self.step_log[k] = [0.0]*len_step_loss + [loss_res['loss_monitor'][k].mean().item()]
                        else:
                            self.step_log[k] =[loss_res['loss_monitor'][k].mean().item()]


                batch_loss += loss.item()
                logging.debug("expname : {0}".format(self.expname))
                logging.debug("timestamp: {0} , epoch : {1}, data_index : {2} , loss : {3} ".format(self.timestamp,
                                                                                                                                                epoch,
                                                                                                                                                data_index,
                                                                                                                                                loss_res['loss'].mean().item()))
                
                for param in self.network.parameters():
                    param.grad = None
                
                data_index = data_index + 1
                
            lr_log_epoch.append(self.optimizer.param_groups[0]["lr"])
            loss_log_epoch.append(batch_loss / (self.ds_len // self.batch_size))
            end = time.time()
            seconds_elapsed_epoch = end - start
            timing_log.append(seconds_elapsed_epoch)
            #logging.debug('end data loop {0}'.format(time.time()-start_epoch))

            if (epoch % self.save_learning_log_freq == 0):
                trace_steploss = []
                selected_stepdata = pd.DataFrame(self.step_log)
                for x in selected_stepdata.columns:
                    if 'loss' in x:
                        trace_steploss.append(
                            go.Scatter(x=np.arange(len(selected_stepdata)), y=selected_stepdata[x], mode='lines',
                                       name=x, visible='legendonly'))

                fig = go.Figure(data=trace_steploss)

                path = os.path.join(self.conf.get_string('train.base_path'), self.exps_folder_name, self.expname, self.timestamp)


                env = '/'.join([path, 'loss'])
                if win is None:
                    win = self.visdomer.plot_plotly(fig, env=env)
                else:
                    self.visdomer.plot_plotly(fig, env=env,win=win)

                self.save_learning_log(epoch_log=dict(epoch=range(self.start_epoch, epoch + 1),
                             loss_epoch=loss_log_epoch,
                             time_elapsed=timing_log,
                             lr_epoch=lr_log_epoch),
                                       step_log=self.step_log)
