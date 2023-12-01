# coding: UTF-8
"""
    @date:  2023.02.21-28  week8-9
    @func:  deformation graph.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.functional as F

import pickle
from scipy.spatial import KDTree
from psbody.mesh import Mesh
from .mesh_sampling import generate_transform_matrices, generate_transform_matrices_coma
from .utils_dg import col, batch_rodrigues
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj

eps = sys.float_info.epsilon # 2.220446049250313e-16


class DeformationGraph(nn.Module):
    
    def __init__(self, vert_number=9648, radius=0.015, k=9, sampling_strategy='qslim'): 
        super().__init__()
        
        self.radius = radius
        self.k = k
        self.max_neigh_num = 40
        self.sampling_strategy = sampling_strategy
        self.one_ring_neigh = []
        self.nodes_idx = None
        self.weights = None
        self.influence_nodes_idx = []
        self.dists = []
        
        self.vert_number = vert_number

    def construct_graph(self, category_name, vertices=None, faces=None):
        
        transform_fp = "transform_{}.pkl".format(category_name)
        if self.sampling_strategy == 'qslim':
            m = Mesh(v=vertices, f=faces)
            if os.path.exists(transform_fp):
                with open(transform_fp, 'rb') as f:
                    tmp = pickle.load(f, encoding='latin1')
                    M, A, D = tmp['M'], tmp['A'], tmp['D']
            else:
                M, A, D = generate_transform_matrices(m, [20, 20])
                tmp = {'M': M, 'A': A, 'D': D}
                with open(transform_fp, 'wb') as fp:
                    pickle.dump(tmp, fp)
            # import pdb; pdb.set_trace()
            nodes_v = M[1].v
            self.nodes_idx = D[0].nonzero()[1]
            adj_mat = A[1].toarray()
            
            for i in range(adj_mat.shape[0]):
                self.one_ring_neigh.append(adj_mat[i].nonzero()[0].tolist() + [i]*(self.max_neigh_num-len(adj_mat[i].nonzero()[0])))
            self.one_ring_neigh = torch.tensor(self.one_ring_neigh).cuda()  

        # construct kd tree
        kdtree = KDTree(nodes_v)
        
        for vert in vertices:
            dist, idx = kdtree.query(vert, k=self.k)
            self.dists.append(dist)
            self.influence_nodes_idx.append(idx)
            
        self.weights = -np.log(np.array(self.dists)+eps)
        
        # weights normalization
        self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).cuda()
        self.influence_nodes_idx = torch.tensor(self.influence_nodes_idx).cuda()
        
    def forward(self, vertices, opt_d_rotations, opt_d_translations):
        
        opt_d_rotmat = batch_rodrigues(opt_d_rotations[0]).unsqueeze(0) # 1 * N_c * 3 * 3
        nodes = vertices[self.nodes_idx, ...]
        
        opt_d_rotmat = opt_d_rotmat.cuda()
        opt_d_translations = opt_d_translations.cuda()

        influence_nodes_v = nodes[self.influence_nodes_idx.reshape((-1,))]# .reshape((28944(self.k * 9648),3,3))
        opt_d_r = opt_d_rotmat[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((28944,3,3,3)) 
        opt_d_t = opt_d_translations[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((28944,3,3))
        
        warpped_vertices = (torch.einsum('bij, bkj->bki', opt_d_r.cuda(), (vertices.repeat_interleave(self.k, dim=0) - influence_nodes_v).unsqueeze(1)).squeeze(1) \
                            + influence_nodes_v + opt_d_t.cuda()).reshape((self.vert_number, self.k, 3)) * (self.weights.unsqueeze(-1))
        warpped_vertices = warpped_vertices.sum(axis=1).float()

        diff_term = (nodes + opt_d_translations[0].cuda()).repeat_interleave(self.max_neigh_num, dim=0) - \
                    (nodes[self.one_ring_neigh.reshape((-1,))] + opt_d_translations[0][self.one_ring_neigh.reshape((-1,))].cuda()) - \
                     torch.einsum('bij, bkj->bki', opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0).cuda(), \
                    (nodes.repeat_interleave(self.max_neigh_num, dim=0) - nodes[self.one_ring_neigh.reshape((-1,))]).unsqueeze(1)).squeeze(1)
        arap_loss = torch.sum(diff_term ** 2) / self.nodes_idx.shape[0]
        
        return warpped_vertices.unsqueeze(0), arap_loss
    
if __name__ == "__main__":
    obj_filename = os.path.join('template/reference/1220/mesh/weiyi.obj')
    verts, faces, aux = load_obj(
                obj_filename,
                device=self.device,
                load_textures=True)
    self.faces = faces.verts_idx
    self.verts = verts