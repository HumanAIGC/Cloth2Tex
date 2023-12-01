# coding: UTF-8
""" 
    deform graph optimization
"""

import cv2
import random

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.transforms import RotateAxisAngle

from utils.handpick_edge import edge_extraction
from utils.mask_iou_loss import mask_iou
from pytorch3d.io import load_obj, save_obj


class DeformGraphModel(torch.nn.Module):
    def __init__(self, deform_graph, renderer, binarization, canonical_mesh, std_lst, lr_rate=5e-4, savedir="1017"):
        super(DeformGraphModel, self).__init__()
        
        self.device = torch.device("cuda:0")
        
        self.deform_graph = deform_graph
        self.cloth_renderer = renderer
        self.binarization = binarization
        self.canonical_mesh = canonical_mesh
        
        self.step_size = lr_rate
        
        self.device = torch.device("cuda:0")
        self.std_lst = std_lst[0]
        self.savedir = savedir
        # self.std_lst_b = std_lst[1]
        
    def iterative_deformgraph(self,
                              batch_id,
                              vertex_number,
                              inputs,
                              contours,
                              verts,
                              opt_d_rotations,
                              opt_d_translations,
                              times=101):
        
        verts_for_dg = verts.detach()
        verts_for_dg.requires_grad = False
        
        surface_optimizer = torch.optim.Adam([
                {'params': [opt_d_rotations]},
                {'params': [opt_d_translations]}
            ], lr=self.step_size)
        
        w_dg = 50
        w_kp = 0.001
        w_lap = 100
        w_norm = 10
        w_arap = 50
        w_edge = 1
        
        min_loss = 10000
        loop = tqdm(range(times))
        
        inputs_front, inputs_back = inputs[0].to(self.device).float(), inputs[1].to(self.device).float()
        landmark_front, landmark_back = contours[0].to(self.device).float(), contours[1].to(self.device).float()   # landmark (2023.02.15)
        
        
        for i in loop:
            surface_optimizer.zero_grad()
            
            # arap: as rigid as possible
            warpped_vertices, loss_arap = self.deform_graph(verts_for_dg, opt_d_rotations, opt_d_translations)
            warpped_vertices = warpped_vertices.squeeze()
            
            src_mesh = Meshes([warpped_vertices], [self.cloth_renderer.faces], self.cloth_renderer.texture)
            
            # front&back
            masks = torch.stack([inputs_front[0], inputs_back[0]]).squeeze()
            
            # mn
            if landmark_back.shape[1] < landmark_front.shape[1]:
                _cc = [landmark_back, torch.zeros(1,1,1,2).cuda()] # original
                # _cc = [landmark_back, torch.zeros(1,1,2).cuda()] # blender
                landmark_back = torch.cat(_cc, 1)
            
            landmarks_canon = torch.stack([landmark_front.squeeze(), landmark_back.squeeze()])
            
            render_mask, specific_verts_2d = self.cloth_renderer.render_silhouette(warpped_vertices, side='both', landmark=True, vertex_number=vertex_number)
            
            # mn
            if specific_verts_2d[0].shape[0] != specific_verts_2d[1].shape[0]:
                _dd = [specific_verts_2d[1], torch.zeros(1,2).cuda()]
                specific_verts_2d[1] = torch.cat(_dd, 0)
            
            render_mask = render_mask[..., 3]
            render_mask_out = self.binarization(render_mask)
            
            loss_dg = nn.MSELoss()(render_mask_out, masks) + 0.3 * mask_iou(render_mask_out, masks) # [2, 512, 512] [2, 512, 512]
            loss_kp = nn.MSELoss()(torch.stack(specific_verts_2d), landmarks_canon)
            edge_mask = edge_extraction(masks)[:, 0].float()
            edge_render_mask = edge_extraction(render_mask_out)[:, 0].float()
            
            loss_edge = nn.L1Loss()(edge_render_mask*render_mask_out, edge_mask)
            
            loss_lap = mesh_laplacian_smoothing(src_mesh, method="uniform")
            loss_norm = mesh_normal_consistency(src_mesh)
            
            # loss = w_dg*loss_dg + w_kp*loss_kp + w_norm*loss_norm + w_arap*loss_arap + w_edge*loss_edge
            loss = w_dg*loss_dg + w_kp*loss_kp + w_norm*loss_norm + w_arap*loss_arap + w_edge*loss_edge # + w_lap*loss_lap + w_norm*loss_norm
            
            loss.backward()
            surface_optimizer.step()
            
            with torch.no_grad():
                render_mask, specific_verts_2d = self.cloth_renderer.render_silhouette(warpped_vertices, side='both', landmark=True, vertex_number=vertex_number)
                f_render_mask, b_render_mask = render_mask[0, ..., 3], render_mask[1, ..., 3]
                f_render_mask, b_render_mask = self.binarization(f_render_mask), self.binarization(b_render_mask)
                
                _f_2d, _b_2d = specific_verts_2d[0].cpu().numpy().copy(), specific_verts_2d[1].cpu().numpy().copy()
                
            loop.set_description('[Total]{0:.2f}[Mask]{1:.2f}[Nor]{2:.2f}[KP]{3:.2f}[ARAP]{4:.2f}[Edge]{5:.2f}'.format(loss, w_dg * loss_dg, w_norm*loss_norm, w_kp*loss_kp, w_arap*loss_arap, w_edge*loss_edge))
            
            if float(loss) < min_loss:
                min_loss = float(loss)
                
                aaa1 = f_render_mask.detach().cpu().numpy() * 255.
                aaa2 = b_render_mask.detach().cpu().numpy() * 255.
                
                bbb1 = inputs_front[0][0].unsqueeze(-1).cpu().numpy() * 255.
                bbb2 = inputs_back[0][0].unsqueeze(-1).cpu().numpy() * 255.
                
                if len(aaa1.shape) == 2:
                    aaa1 = np.expand_dims(aaa1, -1)
                    aaa2 = np.expand_dims(aaa2, -1)
                
                ccc1 = aaa1 * 0.4 + bbb1
                ccc2 = aaa2 * 0.4 + bbb2
                cv2.putText(ccc1, "front", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (193, 33, 240), 2, cv2.LINE_AA) 
                cv2.putText(ccc2, "back", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (193, 33, 240), 2, cv2.LINE_AA) 
                
                for iii, vvvv in enumerate(_f_2d):
                    cv2.circle(ccc1, (int(vvvv[0]), int(vvvv[1])), 3, (193, 33, 240), -1)
                    cv2.putText(ccc1, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 90), 2, cv2.LINE_AA)                    
                for iii, vvvv in enumerate(landmarks_canon[0]):
                    cv2.circle(ccc1, (int(vvvv[0]), int(vvvv[1])), 3, (193, 33, 240), -1)
                    cv2.putText(ccc1, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 40, 200), 2, cv2.LINE_AA) 
                    
                for iii, vvvv in enumerate(_b_2d):
                    if int(vvvv[0]) != 0:
                        cv2.circle(ccc2, (int(vvvv[0]), int(vvvv[1])), 3, (193, 33, 240), -1)
                        cv2.putText(ccc2, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 90), 2, cv2.LINE_AA) 
                
                for iii, vvvv in enumerate(landmarks_canon[1]):
                    if int(vvvv[0]) != 0:
                        cv2.circle(ccc2, (int(vvvv[0]), int(vvvv[1])), 3, (193, 33, 240), -1)
                        cv2.putText(ccc2, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 40, 200), 2, cv2.LINE_AA) 
                
                
                cv2.imwrite("experiments/{0}/{1}_step2_min.jpg".format(self.savedir, batch_id), cv2.hconcat([(ccc1.astype(np.uint8)), ccc2.astype(np.uint8)]))
                
                
                ddd1, ddd2 = edge_render_mask[0].unsqueeze(-1).cpu().numpy() * 255., edge_render_mask[1].unsqueeze(-1).cpu().numpy() * 255.
                cv2.imwrite("experiments/{0}/{1}_step2_edge.jpg".format(self.savedir, batch_id), cv2.hconcat([(ddd1.astype(np.uint8)), ddd2.astype(np.uint8)]))
                
                minimum_vertices = warpped_vertices.clone()
                best_opt_d_rot = opt_d_rotations.clone()
                best_opt_d_trans = opt_d_translations.clone()
            
            # if i >= 50:
            #     if i % 50 == 0:
            #         save_obj("experiments/batch_result/mesh/0505_{}.obj".format(i), warpped_vertices.detach(), self.cloth_renderer.faces)
            # else:
            #     if i % 5 == 0:
            #         save_obj("experiments/batch_result/mesh/0505_{}.obj".format(i), warpped_vertices.detach(), self.cloth_renderer.faces) 

            if i % 500 == 0:
                aaa1 = f_render_mask.detach().cpu().numpy() * 255.
                aaa2 = b_render_mask.detach().cpu().numpy() * 255.
                
                bbb1 = inputs_front[0][0].unsqueeze(-1).cpu().numpy() * 255.
                bbb2 = inputs_back[0][0].unsqueeze(-1).cpu().numpy() * 255.
                
                if len(aaa1.shape) == 2:
                    aaa1 = np.expand_dims(aaa1, -1)
                    aaa2 = np.expand_dims(aaa2, -1)
                
                ccc1 = aaa1 * 0.4 + bbb1
                ccc2 = aaa2 * 0.4 + bbb2
                cv2.putText(ccc1, "front", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (193, 33, 240), 2, cv2.LINE_AA) 
                cv2.putText(ccc2, "back", (int(10), int(40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (193, 33, 240), 2, cv2.LINE_AA) 
                
                for iii, vvvv in enumerate(_f_2d):
                    cv2.circle(ccc1, (int(vvvv[0]), int(vvvv[1])), 3, (80, 40, 200), -1)
                    cv2.putText(ccc1, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 90), 2, cv2.LINE_AA)                    
                for iii, vvvv in enumerate(landmarks_canon[0]):
                    cv2.circle(ccc1, (int(vvvv[0]), int(vvvv[1])), 3, (80, 40, 200), -1)
                    cv2.putText(ccc1, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 40, 200), 2, cv2.LINE_AA) 
                    
                for iii, vvvv in enumerate(_b_2d):
                    if int(vvvv[0]) != 0:
                        cv2.circle(ccc2, (int(vvvv[0]), int(vvvv[1])), 3, (80, 40, 200), -1)
                        cv2.putText(ccc2, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 90), 2, cv2.LINE_AA) 
                
                for iii, vvvv in enumerate(landmarks_canon[1]):
                    if int(vvvv[0]) != 0:
                        cv2.circle(ccc2, (int(vvvv[0]), int(vvvv[1])), 3, (80, 40, 200), -1)
                        cv2.putText(ccc2, self.std_lst[iii], (int(vvvv[0]), int(vvvv[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 40, 200), 2, cv2.LINE_AA) 
                    
                cv2.imwrite("experiments/{0}/{1}_step2_{2}.jpg".format(self.savedir, batch_id, i), cv2.hconcat([(ccc1.astype(np.uint8)), ccc2.astype(np.uint8)]))
        
        
        print("[cloth2tex] [deformation graph parameter]", opt_d_rotations.shape, opt_d_translations.shape)
        return minimum_vertices, best_opt_d_rot, best_opt_d_trans
    
    def forward(self, x):
        out = self.linear(x)
        # out = self.sigmoid(out)
        return out

        
        
if __name__ == "__main__":
    net = LinearModel(2, 2).cuda() # NCHW
    # import pdb; pdb.set_trace()