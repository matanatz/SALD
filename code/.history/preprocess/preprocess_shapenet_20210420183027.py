from __future__ import print_function
import sys
import torch
sys.path.append('../code')
import argparse

import utils.general as utils
import trimesh
from trimesh.sample import sample_surface
import os
import numpy as np
import json

from scipy.spatial import cKDTree
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/home/atzmonm/data/datasets/ShapeNetCore.v2', help="Path of unzipped models.")
    parser.add_argument('--shapeindex', type=int,default=0, help="Shape index to be preprocessed.")
    parser.add_argument('--skip', action="store_true",default=True)
    parser.add_argument('--sigma', type=float,default=0.5)

    args = parser.parse_args()
    split_file = './confs/splits/shapenet/all.json'
    with open(split_file, "r") as f:
        all_models = json.load(f)
    shapeindex = args.shapeindex - 1
    datapath = args.datapath
    global_shape_index = 0
    for cat in [x for x in sorted(os.listdir(os.path.join(datapath))) if not '.' in x and x in all_models['ShapeNet_processed'].keys() and x == '04256520']:

            source = '/home/atzmonm/data/datasets/ShapeNetCore.v2/{0}/'.format(cat)
            output = os.path.join(datapath,'ShapeNet_processed/')
            utils.mkdir_ifnotexists(output)
            utils.mkdir_ifnotexists(os.path.join(output,cat))
            utils.mkdir_ifnotexists(os.path.join(output,cat,'all'))
            shapes = [x for x in sorted(os.listdir(source)) if not '.' in x and x in all_models['ShapeNet_processed'][cat]]

            for shape in shapes:

                if (shapeindex == global_shape_index or shapeindex == -1):
                    print ("found!")
                    output_file = os.path.join(output,cat,'all',shape)
                    print (output_file)
                    if (not args.skip or  not os.path.isfile(output_file + '_dist_triangle.npy')):

                        print ('loading : {0}'.format(os.path.join(source,shape)))
                        mesh = trimesh.load(os.path.join(source,shape) + '/models/model_normalized.obj')
                        mesh = as_mesh(mesh)
                        sample = sample_surface(mesh,250000)
                        center = np.mean(sample[0],axis=0)

                        pnts = sample[0]

                        pnts = pnts - np.expand_dims(center,axis=0)
                        scale = np.abs(pnts).max()
                        pnts = pnts/scale
                        triangles = []
                        i = 0
                        for tri in mesh.triangles:
                            
                            a = Point_3((tri[0][0] - center[0])/scale, (tri[0][1]- center[1])/scale, (tri[0][2]- center[2])/scale)
                            b = Point_3((tri[1][0]- center[0])/scale, (tri[1][1]- center[1])/scale, (tri[1][2]- center[2])/scale)
                            c = Point_3((tri[2][0]- center[0])/scale, (tri[2][1]- center[1])/scale, (tri[2][2]- center[2])/scale)
                            triangles.append(Triangle_3(a, b, c))
                            
                            i = i +1
                        tree = AABB_tree_Triangle_3_soup(triangles)
                        print ('after traingles')
                        sigmas = []
                        ptree = cKDTree(pnts[np.random.choice(np.arange(pnts.shape[0]),10000,False)])
                        i = 0
                        for p in np.array_split(pnts,100,axis=0):
                            d = ptree.query(p,51)
                            sigmas.append(d[0][:,-1])

                            i = i+1
                            
                        sigmas = np.concatenate(sigmas)
                        sigmas_big = args.sigma * np.ones_like(sigmas)

                        np.save(output_file + '.npy', np.concatenate([pnts, mesh.face_normals[sample[1]]],axis=-1))


                        sample = np.concatenate([pnts + np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape),
                                         pnts + np.expand_dims(sigmas_big,-1) * np.random.normal(0.0,1.0, size=pnts.shape)], axis=0)

                        dists = []
                        normals = []
                        i = 0
                        for np_query in sample:
                            cgal_query = Point_3(np_query[0].astype(np.double), np_query[1].astype(np.double), np_query[2].astype(np.double))

                            cp = tree.closest_point(cgal_query)
                            cp = np.array([cp.x(), cp.y(), cp.z()])
                            dist = np.sqrt(((cp - np_query)**2).sum(axis=0))
                            n = (cp - np_query) / dist
                            normals.append(np.expand_dims(n.squeeze(), axis=0))
                            dists.append(dist)
                            
                            i =i+1
                        dists = np.array(dists)
                        normals = np.concatenate(normals, axis=0)
                        np.save(output_file + '_dist_triangle.npy',
                                np.concatenate([sample,normals, np.expand_dims(dists, axis=-1)], axis=-1))

                        np.save(output_file + '_normalization.npy',
                                {"center":center,"scale":scale})

                global_shape_index = global_shape_index + 1

    print ("end!")