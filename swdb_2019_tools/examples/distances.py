#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:12:56 2019

@author: baharrahsepar
"""
# this is the EM specific package for querying the EM data
import platform
import os
from analysisdatalink.datalink_ext import AnalysisDataLinkExt as AnalysisDataLink
# import some of our favorite packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Modules from meshparty
from meshparty import mesh_skel_utils
from meshparty import trimesh_io, trimesh_vtk
from meshparty import skeletonize, skeleton_io, skeleton

import scipy
from meshparty import mesh_filters

#%%
def sk_dist(neuronID, pts, data_root, dataset_name='pinky100', filt=True,max_dist=2000):

        
    '''
    Calculates distance of any point/points in the space to the closest point on the skeleton neuron of interest
    
    INPUTS:
        neuronID                    = String, id associated with the neuron of interest
        pts                         = np.array 3xN, N: number of points
        data_root                   = Location of the dataset  
        datset_name (optional)      = string,(defaul pinky100)
        filt (optional)             = bool, filter the mesh based on segment size(default True)
        max_dist (optional)         = float, maximum expected distance from the neuron in nm (defaul 500nm)
    
    RETURNS: 
        dists              = Nx1 np array, distances of each point to soma in nm
    '''
    
    
    # set values
    voxel_size = [4,4,40]
    # Folders for the mesh and skeleton
    mesh_folder = os.path.join(data_root, 'meshes')
    skeleton_folder = os.path.join(data_root, 'skeletons')
    # Mesh meta data
    mm = trimesh_io.MeshMeta(cv_path = 'graphene://https://swdb.dynamicannotationframework.com/segmentation/1.0/pinky100_sv16',
                         disk_cache_path=mesh_folder,
                         cache_size=2)
    
    # load the mesh for the neuron
    mesh = mm.mesh(seg_id = neuronID) 
    #load the skeleton for the neuron
    sk=skeleton_io.read_skeleton_h5(skeleton_folder+'/'+str(neuronID)+'.h5')
    
    if filt:
        # filter out the segmented portions of the mesh
        mask = mesh_filters.filter_largest_component(mesh)
        neuron_mesh = mesh.apply_mask(mask)
    else:
        neuron_mesh=mesh
    
    #load the skeleton for the neuron
    sk=skeleton_io.read_skeleton_h5(skeleton_folder+'/'+str(neuron)+'.h5')

    #convert vertecies to nm
    pt_nm=np.vstack(pts)*np.array(voxel_size)
    
    # use kdtree to find the shortest distance from the point to the mesh and the index associated with that
    dist,ind=neuron_mesh.kdtree.query(pt_nm,distance_upper_bound=max_dist)
    if filt:
        #find the index on original mask
        ind=neuron_mesh.map_indices_to_unmasked(ind_masked)
        neuron_mesh=mesh


    #find skeleton vertex of the point on the mesh
    syn_sk_ind=sk.mesh_to_skel_map[ind]
    syn_sk_mesh_ind=np.array(sk.vertex_properties['mesh_index'])[syn_sk_ind]

    dd=scipy.sparse.csgraph.dijkstra(neuron_mesh.csgraph, directed=False, 
                                     indices=ind)

    dists=[dd[ind,mesh_ind] for ind,mesh_ind in enumerate(syn_sk_mesh_ind)]
    dists=np.array(dists)
    return dists



#%%
def soma_dist(neuronID, pts, data_root, dataset_name='pinky100', filt=True,max_dist=500):
    
    '''
    Calculates distance of any point/points in the space to the center of the soma of neuron of interest
    
    INPUTS:
        neuronID                    = String, id associated with the neuron of interest
        pts                         = np.array 3xN, N: number of points
        data_root                   = Location of the dataset  
        datset_name (optional)      = string,(defaul pinky100)
        filt (optional)             = bool, filter the mesh based on segment size(default True)
        max_dist (optional)         = float, maximum expected distance from the neuron in nm (defaul 500nm)
    
    RETURNS: 
        pt_soma_dist              = Nx1 np array, distances of each point to soma in nm
    '''
    
    
    # set values
    voxel_size = [4,4,40]
    # Folders for the mesh and skeleton
    mesh_folder = os.path.join(data_root, 'meshes')
    skeleton_folder = os.path.join(data_root, 'skeletons')
    # Mesh meta data
    mm = trimesh_io.MeshMeta(cv_path = 'graphene://https://swdb.dynamicannotationframework.com/segmentation/1.0/pinky100_sv16',
                         disk_cache_path=mesh_folder,
                         cache_size=2)
    
    # load the mesh for the neuron
    mesh = mm.mesh(seg_id = neuronID) 
    #load the skeleton for the neuron
    sk=skeleton_io.read_skeleton_h5(skeleton_folder+'/'+str(neuronID)+'.h5')
    
    if filt:
        # filter out the segmented portions of the mesh
        mask = mesh_filters.filter_largest_component(mesh)
        neuron_mesh = mesh.apply_mask(mask)
    else:
        neuron_mesh=mesh
    
        
    # convert vertecies to nm    
    pt_nm=np.vstack(pts)*np.array(voxel_size)
    # use kdtree to find the shortest distance from the point to the mesh and the index associated with that
    dist,ind=neuron_mesh.kdtree.query(pt_nm,distance_upper_bound=500)

    # use kdtree to find the shortest distance from the point to the mesh and the index associated with that
    dist,ind=neuron_mesh.kdtree.query(pt_nm,distance_upper_bound=max_dist)
    
    #find the vertices of the synapse point on the mesh 
    pt_pos_mesh=neuron_mesh.vertices[ind]
    
    #find skeleton vertex of the point on the mesh
    if filt:
        ind_orig=neuron_mesh.map_indices_to_unmasked(ind)
        pt_sk_vert=sk.mesh_to_skel_map[ind_orig]
    else:
        pt_sk_vert=sk.mesh_to_skel_map[ind]
    
    
    pt_soma_dist=sk.distance_to_root[pt_sk_vert]+dist
    +sk_dist(neuronID, pts, data_root, dataset_name='pinky100', filt=True,max_dist=2000)
    return pt_soma_dist