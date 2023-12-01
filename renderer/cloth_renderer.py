# coding: UTF-8

"""
    clothrenderer
"""


import datetime

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
import torchvision.transforms as transforms
from PIL import Image
import random

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    BlendParams,
    OrthographicCameras,
    FoVOrthographicCameras,
    FoVPerspectiveCameras, 
    PointLights,
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    TexturesUV
)

from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.transforms import RotateAxisAngle, Rotate, axis_angle_to_matrix

from renderer.landmark_renderer import PerspectiveCamera, OrthogonalCamera

DEG_TO_RAD = np.pi / 180


class ClothRenderer(object):
    
    def __init__(self, objfile, resolution=512, focal_distance=1.6, scale_factor=1):
        self.device = torch.device("cuda:0")

        self.img_size = resolution
        self.render_size = resolution
        self.renderer, self.renderer_silhouette = self.__get_renderer(self.render_size, focal_distance)
        
        print("[Cloth2Tex]", objfile)
        obj_filename = os.path.join(objfile)
        verts, faces, aux = load_obj(
                    obj_filename,
                    device=self.device,
                    load_textures=True)
        self.faces = faces.verts_idx
        self.verts = verts
        self.aux = aux
        
        self.verts = self.normalize_vertex(verts.clone()) * scale_factor
        
        self.center = verts.mean(0)
        self.scale = max((verts - self.center).abs().max(0)[0])
        self.landmark_cam = OrthogonalCamera(rotation=self.cameras.R.cuda(), translation=self.cameras.T.cuda()).to(self.device)
        
        _keys = []
        if len(aux.texture_images.keys()) > 0:
            for _ in aux.texture_images.keys():
                _keys.append(_)
            self.tex_lst = [aux.texture_images[i] for i in _keys]
            texture_image = self.tex_lst[0]
        
        
        self.verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        tex_maps = aux.texture_images

        # Canonical Mesh
        texture_image = texture_image[None, ...].to(self.device)  # (1, H, W, 3)
        self.texture = TexturesUV(maps=texture_image, faces_uvs=self.faces[None], verts_uvs=self.verts_uvs)
        self.canonical_mesh = Meshes([self.verts], [self.faces], self.texture)
    
    def normalize_vertex(self, verts):
        # Normalizing
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        
        verts = verts - center
        verts = verts * (1/float(scale))
        
        return verts
    
    def denormalize_vertex(self, verts):
        
        out = self.scale*verts + self.center
        
        return out
    
    def render_silhouette(self, verts, side='back', landmark=True, vertex_number=[[], []]):
        vert_lst_front = vertex_number[0]
        vert_lst_back = vertex_number[1]
            
        tmp_verts = verts.clone()
        mesh = Meshes([tmp_verts], [self.faces], self.texture)
        meshes = mesh.extend(2)
        
        # Get a batch(2) of viewing angles. 
        elev = torch.linspace(180, -180, 2)
        azim = torch.linspace(0, 0, 2)
        
        focal_length = torch.linspace(-1, 1, 2)
        R, T = look_at_view_transform(dist=focal_length, elev=elev, azim=azim)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)
        
        target_images, fragments = self.renderer_silhouette(meshes, cameras=cameras)
           
        if landmark is True:
            # project normalized vertex to image space(fix vertex)
            specific_verts_2d_front = self.landmark_cam(verts[vert_lst_front].unsqueeze(0))[0]
            # conversion from OpenGL coordinate to OpenCV coordinate
            specific_verts_2d_front[:,] = -specific_verts_2d_front[:,]
            # conversion from [-1,1] to [0,512]
            specific_verts_2d_front = (specific_verts_2d_front+1)/2*self.render_size
            
            # project normalized vertex to image space(fix vertex)
            specific_verts_2d_back = self.landmark_cam(verts[vert_lst_back].unsqueeze(0))[0]
            # conversion from OpenGL coordinate to OpenCV coordinate
            specific_verts_2d_back[:,] = -specific_verts_2d_back[:,]
            # conversion from [-1,1] to [0,512]
            specific_verts_2d_back = (specific_verts_2d_back+1)/2*self.render_size
            
            if side == 'front':
                return target_images[0], [specific_verts_2d_front]
            elif side == 'back':
                return target_images[1], [specific_verts_2d_back]
            else:
                return target_images, [specific_verts_2d_front, specific_verts_2d_back]
        
        return target_images, fragments
    
    def render_image(self, texture_image):
        texture = TexturesUV(maps=texture_image, faces_uvs=self.faces[None], verts_uvs=self.verts_uvs)
        
        tmp_verts = self.verts.clone()
        mesh = Meshes([tmp_verts], [self.faces.clone()], texture)
        meshes = mesh.extend(2)
        
        # Get a batch(2) of viewing angles. 
        elev = torch.linspace(180, -180, 2)
        azim = torch.linspace(0, 0, 2)
        
        focal_length = torch.linspace(-1, 1, 2)
        R, T = look_at_view_transform(dist=focal_length, elev=elev, azim=azim)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)
        
        target_images = self.renderer(meshes, cameras=cameras)
        target_masks, _ = self.renderer_silhouette(meshes, cameras=cameras)
        
        return target_images, target_masks
    
    
    def __get_renderer(self, render_size, focal_distance=2):
        
        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]],
                             ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
        
        self.focal_distance = focal_distance
        R, T = look_at_view_transform(focal_distance, -180, 0) # 180 -> -180
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T) # silhouette only！
        # cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)
        
        self.cameras = cameras
        
        raster_settings = RasterizationSettings(
            image_size=render_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        sigma = 1e-4
        gamma = 1e-4
        blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(255, 255, 255))
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader = SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                # blend_params=blend_params
            )
        )
        
        # ref: https://github.com/facebookresearch/pytorch3d/issues/470
        sigma = 1e-8
        gamma = 1e-8
        blend_params = BlendParams(sigma=sigma, gamma=gamma, background_color=(0, 0, 0))
        raster_settings = RasterizationSettings(
            image_size=render_size, 
            blur_radius=np.log(1. / 1e-8 - 1.)*sigma, # blur_radius=np.log(1. / 1e-8 - 1.)*sigma, 
            faces_per_pixel=10, 
            bin_size=None, 
            max_faces_per_bin=None
        )
        
        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
            # shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        return renderer, renderer_silhouette
    


if __name__ == "__main__":
    cloth = ClothRenderer()
