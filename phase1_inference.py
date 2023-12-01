# -*- coding: utf-8 -*-

"""
    @date:  2023.03.29-31  week13
    @func:  PhaseI inference code.
"""

import argparse
import datetime

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import os.path as osp

import torchvision
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

import torch.nn.functional as F
import thinplate as tps
import time
from renderer.cloth_renderer import ClothRenderer
import matplotlib.pyplot as plt
from PIL import Image
import importlib
import random
from utils.frequency import extract_ampl_phase
from utils.binary_function import Binarize
from utils.tvl_loss import TVLoss, TVMaskLoss

from tqdm import tqdm
import json
from pytorch3d.io import load_obj, save_obj
import cv2
from itertools import chain

from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from lib.deformation_graph import DeformationGraph
from lib.mesh_sampling import generate_transform_matrices_coma
from lib.utils_dg import to_edge_index, to_sparse, get_vert_connectivity, scipy_to_torch_sparse
from models import DeformGraphModel

from torch_geometric.transforms import FaceToEdge
from torch_geometric.data import Data


from psbody.mesh import Mesh
from torch_geometric.io import read_ply


class Trainer(object):
    def __init__(self, objfile, savedir, resolution=512, focal_distance=2, verts_num=9648, scale_factor=1.0):
        
        self.device = torch.device("cuda")
        #set mesh and visualizer----------------------
        self.cloth_renderer = ClothRenderer(objfile, resolution, focal_distance, scale_factor)
        
        
        if os.path.exists(os.path.join("experiments", savedir)):
            pass
        else:
            os.makedirs(os.path.join("experiments", savedir))
        
        self.savedir = savedir
        
        self.uv = torch.ones((1, 512, 512, 3)).cuda()
        self.uv.requires_grad = True
        self.optimizer = optim.Adam([self.uv], lr=5e-3, betas=(0.5, 0.999))
        
        # define loss
        self.criterion = nn.MSELoss() # nn.L1Loss() nn.MSELoss()
        self.mse = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # You can choose TVMaskLoss and test if it is suitable for your case.
        self.tvl_loss = TVLoss(weight=1) # TVMaskLoss(weight=1) or TVLoss(weight=1)
        # self.tvl_loss = TVMaskLoss(weight=1)
        
        self.canonical_mesh = self.cloth_renderer.canonical_mesh
        self.deform_verts = self.cloth_renderer.verts.to(self.device)
        self.deform_verts.requires_grad = False
        
        self.deform_graph = DeformationGraph(vert_number=verts_num)
        
        self.mask_silhouette = None
        self.w_photo = 100
        self.w_tex = 10
        self.w_perp = 2
        self.w_ssim = 1
        self.w_tvl = 1
        
        _tmp_model1 = torchvision.models.resnext50_32x4d(pretrained=True).to(self.device)
        self.feature_extractor_coarse = create_feature_extractor(_tmp_model1, {'layer1': 'feat1', 'layer4': 'feat2'})
        
        _tmp_model2 = torchvision.models.vgg16(pretrained=True).to(self.device)
        self.feature_extractor_fine = create_feature_extractor(_tmp_model2, {'features.12': 'feat1', 'features.30': 'feat2'})
        
        self.binarization = Binarize.apply
        self.face2edge = FaceToEdge(remove_faces=False)
    
    def create_graph(self, garment_type, verts, faces, std_lst):
        print("Start Graph Creation ...")
        self.deform_graph.construct_graph(garment_type, verts.cpu(), faces.cpu())
        
        num_nodes = self.deform_graph.nodes_idx.shape[0]
        self.opt_d_rotations = torch.nn.Parameter(torch.zeros(1, num_nodes, 3), requires_grad=True) # axis angle 
        self.opt_d_translations = torch.nn.Parameter(torch.zeros(1, num_nodes, 3), requires_grad=True)
        
        self.dg = DeformGraphModel(deform_graph=self.deform_graph,
                                   renderer=self.cloth_renderer,
                                   binarization=self.binarization,
                                   canonical_mesh=self.canonical_mesh,
                                   std_lst=std_lst,
                                   lr_rate=1e-2,
                                   savedir=self.savedir)
        print("Finish Graph Creation !!!")
        
        
    def iterative_mesh(self,
                       garment_type,
                       batch_id,
                       std_lst,
                       vertex_number,
                       inputs,
                       contours,
                       times=1001):
        
        # Weight for mask & keypoint
        w_mask = 10.0
        w_kp = 0.005
        # Weight for mesh edge loss
        w_edge = 1.0 
        # Weight for mesh normal consistency
        w_normal = 0.01 
        w_normal = 0.1
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.1
        
        loop = tqdm(range(401))
        
        inputs_front, inputs_back = inputs[0].to(self.device).float(), inputs[1].to(self.device).float()
        landmark_front, landmark_back= contours[0].to(self.device).float(), contours[1].to(self.device).float()   # landmark (2023.02.15)
        
        # A1 step1:  Align top neckline edges.
        for i in loop:
            render_mask, specific_verts_2d = self.cloth_renderer.render_silhouette(self.deform_verts, side='back', landmark=True, vertex_number=vertex_number)
            render_mask = render_mask[..., 3]
            render_mask = torch.clip(render_mask * 2, 0, 1)
            
            masks = inputs_back[0]
            
            distance = specific_verts_2d[0][0].detach() - landmark_back.squeeze()[0] # 对齐领口左侧.
            
            # image plane up&down
            if abs(distance[1]) > 1:
                if distance[1] > 0:
                    self.deform_verts = self.deform_verts + 0.003 * torch.tensor([0, 1, 0]).to(self.deform_verts.device)
                else:
                    self.deform_verts = self.deform_verts - 0.003 * torch.tensor([0, 1, 0]).to(self.deform_verts.device)
            
            if i % 200 == 0:
                aaa = render_mask.unsqueeze(-1).detach().cpu().numpy() * 255.
                bbb = masks[0].unsqueeze(-1).cpu().numpy() * 255.
                ccc = inputs_front[0][0].unsqueeze(-1).cpu().numpy() * 255.
                cv2.imwrite("experiments/{0}/iterative_{1}_step1_{2}.jpg".format(self.savedir, batch_id, i), cv2.hconcat([(aaa.astype(np.uint8)), bbb.astype(np.uint8), ccc.astype(np.uint8)]))
    
        
        # A2 step2:  deformation graph.
        self.cloth_renderer.verts = self.deform_verts.detach()
        
        _verts = self.deform_verts.detach()
        _verts.requires_grad = False
        _faces = self.cloth_renderer.faces
        _faces.requires_grad = False
        
        self.create_graph(garment_type, _verts, _faces, std_lst)
        final_vertices, opt_d_rot, opt_d_trans = self.dg.iterative_deformgraph(
                                                       batch_id,
                                                       vertex_number,
                                                       inputs,
                                                       contours,
                                                       self.deform_verts.detach(),
                                                       self.opt_d_rotations,
                                                       self.opt_d_translations,
                                                       times=1001)
        print("deformation graph finish!")
        self.cloth_renderer.verts = final_vertices.detach()
        
    def iterative_optimize(self,
                           garment_mask,
                           batch_id,
                           vertex_number,
                           inputs,
                           input_masks,
                           times=1001):
        
        inputs_front, inputs_back = inputs[0].to(self.device).float(), inputs[1].to(self.device).float()
        masks_front, masks_back = input_masks[0].to(self.device).float(), input_masks[1].to(self.device).float()
        
        
        loop = tqdm(range(times))
        for n in loop:
            t1 = time.time()
            self.optimizer.zero_grad()
            
            #=======================Finetune the UV map=====================
            rendered_imgs, rendered_masks = self.cloth_renderer.render_image(self.uv[0].unsqueeze(0))
            #==============================  Loss Definition ===========================================
            outputs = rendered_imgs[:, :, :, :3].permute(0, 3, 1, 2).contiguous()
            render_mask = rendered_masks[..., 3]
            render_mask_out = self.binarization(render_mask)

            # 1) Photometric Loss
            l_photo = nn.L1Loss()(outputs[0].unsqueeze(0), inputs_front) + nn.L1Loss()(outputs[1].unsqueeze(0), inputs_back)
            l_perp = 0.0

            # 2) TVL Loss
            l_tvl = self.tvl_loss(self.uv[0].unsqueeze(0).permute(0, 3, 1, 2))
            loss = self.w_photo*l_photo + self.w_tvl*l_tvl

            #=============================== Backward and optimization =======================================            
            loss.backward()
            self.optimizer.step()
            
            if n % 500 == 0:
                with torch.no_grad():
                    rendered_imgs_all, rendered_masks_all = self.cloth_renderer.render_image(self.uv[0].unsqueeze(0))
                    rendered_imgs_all_finish = rendered_imgs_all[:, :, :, :3].permute(0, 3, 1, 2).contiguous()
                    
                    mask_front_r, mask_back_r = rendered_masks_all[0, ..., 3], rendered_masks_all[1, ..., 3]
                    mask_front_r, mask_back_r = self.binarization(mask_front_r), self.binarization(mask_back_r)
                    
                    rendered_imgs_front, rendered_imgs_back = rendered_imgs_all_finish[0].unsqueeze(0), rendered_imgs_all_finish[1].unsqueeze(0), 
                    
                    
                out_front = (np.clip(rendered_imgs_front.detach().cpu()[0].permute(1,2,0).numpy(), 0.0, 1.0) * 255.)
                gaga_front = inputs_front[0].cpu().permute(1,2,0).numpy() * 255. # * mask_front_r.permute(1,2,0).cpu().numpy()
                
                out_back = (np.clip(rendered_imgs_back.detach().cpu()[0].permute(1,2,0).numpy(), 0.0, 1.0) * 255.)
                gaga_back = inputs_back[0].cpu().permute(1,2,0).numpy() * 255. # * mask_back_r.permute(1,2,0).cpu().numpy()
                
                cv2.imwrite("experiments/{0}/{1}_texture_{2}.jpg".format(self.savedir, batch_id, n), cv2.hconcat([gaga_front.astype(np.uint8), out_front.astype(np.uint8), gaga_back.astype(np.uint8), out_back.astype(np.uint8)]))
                
                out_uv = np.clip(self.uv[0].detach().cpu().numpy()[:, :, :], 0.0, 1.0)
                out_uv = out_uv / out_uv.max()
                out_uv = (out_uv * 255.)
                if garment_mask is not None:
                    garment_mask_img = cv2.imread(garment_mask)
                    cv2.imwrite("experiments/{0}/{1}_texture_uv_{2}.jpg".format(self.savedir, batch_id, n), (out_uv * garment_mask_img / 255. + (255 - garment_mask_img)).astype(np.uint8))
                
                
            loop.set_description("iter {0}, loss: l_photo {1:.3f}, l_tvl: {2:.3f}".format(n, self.w_photo*l_photo, self.w_tvl*l_tvl))
        
        print('Finished Phase I Optimization')
        torch.cuda.empty_cache()

def main(category,
         savedir,
         scale,
         steps_one,
         steps_two):
    img_transform = transforms.ToTensor()
    
    
    category = category
    savedir = savedir
    scale = scale
    steps_one = steps_one
    steps_two = steps_two
    
    mapping_dict = {"1_wy": "wy.jpg",
                    "2_Polo": "polo.jpg",
                    "3_Tshirt": "Tshirt.jpg",
                    "4_shorts": "shorts.jpg",
                    "5_trousers": "trousers.jpg",
                    "6_zipup": "zipup.jpg",
                    "7_windcoat": "windcoat.jpg",
                    "9_jacket": "jacket.jpg",
                    "11_skirt": "skirt.jpg"}
    garment_mask = "template/uv_mask/{}".format(mapping_dict.get(category))
    
    landmark_order_dict = {"1_wy": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "2_Polo": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "3_Tshirt": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "4_shorts": ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],
                           "5_trousers": ['lw', 'mw', 'rw', 'cc', 'llo', 'lli', 'rli', 'rlo'],
                           "6_zipup": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "7_windcoat": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "9_jacket": ['ln', 'rn', 'lh', 'rh', 'lso', 'rso', 'lpo', 'lpi', 'rpo', 'rpi'],
                           "11_skirt": ['lw', 'mw', 'rw', 'll', 'ml', 'rl']}
    per_vertex_dict = {"1_wy": 9648,
                       "2_Polo": 8922,
                       "3_Tshirt": 8523,
                       "4_shorts": 8767,
                       "5_trousers": 9323,
                       "6_zipup": 8537,
                       "7_windcoat": 9881,
                       "9_jacket": 8168,
                       "11_skirt": 6116}
    # NOTICE: we do not release auto scaling code here and release a pre-defined scale coefficient parameters.
    per_scale_dict = {"1_wy": 1.1,
                       "2_Polo": 0.8, # default 0.8
                       "3_Tshirt": 0.9, # default 0.7
                       "4_shorts": 0.75, # # default 0.7
                       "5_trousers": 0.75,
                       "6_zipup": 1.1,
                       "7_windcoat": 0.65,
                       "9_jacket": 1.0,
                       "11_skirt": 1.0} 
    
    
    vertex_number_dict = {"1_wy": [[9613, 9628, 9310, 9341, 5829, 5701, 8916, 8899, 8772, 8755, 9325, 9583],
                                     [9572, 9558, 9004, 8974, 5829, 5701, 8916, 8899, 8772, 8755, 8990]],
                          "2_Polo": [[4318, 4332, 1031, 849, 8485, 91, 8812, 8794, 8871, 8858, 1050, 24], 
                                     [4321, 4330, 4710, 4845, 4759, 4796, 8812, 8794, 8871, 8858, 4691]],
                          "3_Tshirt": [[4427, 658, 4497, 4526, 7828, 8190, 40, 7799, 324, 333, 4511, 4415],
                                       [842, 856, 762, 728, 7830, 8192, 7814, 7802, 8172, 8164, 744]],
                          "4_shorts": [[8215, 8198, 8179, 1772, 5092, 5000, 3398, 3490],
                                       [7639, 7657, 7673, 5018, 5092, 5000, 3398, 3490]],
                          "5_trousers": [[8502, 9081, 8536, 6639, 4384, 4404, 2476, 2331],
                                       [8502, 8518, 8536, 4454, 6538, 6685, 2476, 2331]],
                          "6_zipup": [[3325, 3210, 8305, 8335, 65, 15, 8019, 8012, 8494, 8487, 8319],
                                       [3325, 3211, 8305, 8335, 65, 15, 8019, 8012, 8494, 8487, 8076]],
                          "7_windcoat": [[2378, 2405, 7177, 5531, 8806, 9411, 8835, 8771, 9380, 9372, 5514],
                                         [2378, 2405, 7177, 5532, 8806, 9411, 8835, 8771, 9380, 9372, 2282]],
                          "9_jacket": [[7488, 8022, 7892, 7380, 669, 31, 8120, 8136, 7840, 7856, 5506, 8077],
                                       [7488, 7498, 7892, 7380, 2514, 31, 5436, 5414, 5347, 5325, 7587]],
                          "11_skirt": [[216, 206, 193, 3301, 3348, 507],
                                       [216, 45, 193, 3301, 556, 507]]}
    
    for idx in range(0,2):
        
        trainer = Trainer(objfile="template/reference/{}/mesh/mesh.obj".format(category),
                          savedir=savedir,
                          resolution=512, 
                          focal_distance=1.7,
                          verts_num=per_vertex_dict[category],
                          scale_factor=scale)
        
        if os.path.exists("template/reference/{0}/square/{1}_1.jpg".format(category, idx)) is False:
            continue
        
        ref_img = cv2.imread("template/reference/{0}/bg/{1}_1.jpg".format(category, idx))
        ref_img_back = cv2.imread("template/reference/{0}/bg/{1}_2.jpg".format(category, idx))

        H, W = ref_img.shape[:2]
        H1, W1 = ref_img_back.shape[:2]

        with open("template/reference/{0}/square/{1}_1.json".format(category, idx)) as f:
            result_kp1=json.load(f)

        with open("template/reference/{0}/square/{1}_2.json".format(category, idx)) as f:
            result_kp2=json.load(f)

        std_lst = landmark_order_dict[category].copy()
        tmp_count_f, tmp_count_b = len(result_kp1['shapes']), len(result_kp2['shapes'])
        
        c_src_front = [result_kp1['shapes'][_]['points'][:2] for _ in range(tmp_count_f)]
        c_src_front = np.array(c_src_front)
        c_src_front_label = [result_kp1['shapes'][_]['label'] for _ in range(tmp_count_f)]

        c_src_back = [result_kp2['shapes'][_]['points'][:2] for _ in range(tmp_count_b)]
        c_src_back = np.array(c_src_back)
        c_src_back_label = [result_kp2['shapes'][_]['label'] for _ in range(tmp_count_b)]

        c_src_idx, c_dst_idx = [], []
        print("#"*30)
        print(idx, std_lst)
        print("#"*30)
        for tmp_idx in std_lst:
            c_src_idx.append(c_src_front_label.index(tmp_idx))
            c_dst_idx.append(c_src_back_label.index(tmp_idx))

        c_src_front = c_src_front[c_src_idx]
        c_src_back = c_src_back[c_dst_idx]
        
        # add mh point.
        if "cc" not in std_lst and "skirt" not in category:
            std_lst.append("mh")
            tmp_mh = np.expand_dims((c_src_front[2] + c_src_front[3]) / 2, 0)
            c_src_front = np.append(c_src_front, tmp_mh, axis=0)

            tmp_mh = np.expand_dims((c_src_back[2] + c_src_back[3]) / 2, 0)
            c_src_back = np.append(c_src_back, tmp_mh, axis=0)
            
        # wy, polo, T needs mn
        std_lst_front = std_lst.copy()
        std_lst_back = std_lst.copy()
        if category in ["1_wy", "2_Polo", "3_Tshirt", "9_jacket"]:
            for mn_idx in range(tmp_count_f):
                if result_kp1['shapes'][mn_idx]['label'] == "mn" or result_kp1['shapes'][mn_idx]['label'] == "lmn":
                    tmp_mn = np.expand_dims(np.asarray(result_kp1['shapes'][mn_idx]['points'][:2]), 0)
                    c_src_front = np.append(c_src_front, tmp_mn, axis=0)
                    std_lst_front.append("mn")
                    print("[mn] appending success!")
                    break
        

        inputs = img_transform(ref_img).unsqueeze(0)
        inputs_back = img_transform(ref_img_back).unsqueeze(0)

        c_src_front, c_src_back = torch.from_numpy(c_src_front).unsqueeze(0), torch.from_numpy(c_src_back).unsqueeze(0)

        mask_front = cv2.imread("template/reference/{0}/mask/{1}_1.jpg".format(category, idx), 0)
        mask_back = cv2.imread("template/reference/{0}/mask/{1}_2.jpg".format(category, idx), 0)
        mask_front, mask_back = np.where(mask_front > 10, 255, 0), np.where(mask_back > 10, 255, 0)
        mask_front, mask_back = img_transform(mask_front).unsqueeze(0), img_transform(mask_back).unsqueeze(0)
        mask_front, mask_back = mask_front / 255., mask_back / 255.
        
        # optimize
        trainer.iterative_mesh(category,
                               idx,
                               [std_lst_front, std_lst_back],
                               vertex_number_dict[category],
                               [mask_front, mask_back],
                               [c_src_front, c_src_back],
                               times=steps_one)

        trainer.iterative_optimize(garment_mask,
                                   idx,
                                   vertex_number_dict[category],
                                   [inputs, inputs_back], 
                                   [mask_front, mask_back],
                                   times=steps_two)
        break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g', '--garment', help = 'garment type: 1_wy, 2_Polo, 3_Tshirt, 4_shorts, 5_trousers, 6_zipup, 7_windcoat, 9_jacket, 11_skirt', type = str, default = "1_wy")
    parser.add_argument('--d', '--dstdir', help = 'dst dir', type = str, default = "{}".format(datetime.date.today().strftime("%Y-%m-%d")))
    parser.add_argument('--s', '--scale', help = 'scale coefficient', type = float, default = 1.0)
    parser.add_argument('--steps_one', help = 'optimizer 1 steps', type = int, default = 501)
    parser.add_argument('--steps_two', help = 'optimizer 2 steps', type = int, default = 1001)
    args = parser.parse_args()
    print(args)
    main(args.g, args.d, args.s, args.steps_one, args.steps_two)
