# coding: UTF-8


import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    FoVOrthographicCameras,
    FoVPerspectiveCameras, 
    PointLights, 
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
from pytorch3d.transforms import RotateAxisAngle


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


class OrthogonalCamera(nn.Module):

    def __init__(self, 
                 rotation=None, 
                 translation=None,
                 batch_size=1,
                 center=None, 
                 dtype=torch.float32):
        super(OrthogonalCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)
        self.register_buffer('center_fix', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)
    

    def forward(self, points):
        device = points.device
        
        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = 1
            camera_mat[:, 1, 1] = 1

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])
        img_points = projected_points[:, :, :2]
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)

        return img_points


class PerspectiveCamera(nn.Module):

    FOCAL_LENGTH = 50*128

    def __init__(self, rotation=None, translation=None,
                 focal_length_x=None, focal_length_y=None,
                 batch_size=1,
                 center=None, dtype=torch.float32):
        super(PerspectiveCamera, self).__init__()
        self.batch_size = batch_size
        self.dtype = dtype
        # Make a buffer so that PyTorch does not complain when creating
        # the camera matrix
        self.register_buffer('zero',
                             torch.zeros([batch_size], dtype=dtype))

        if focal_length_x is None or type(focal_length_x) == float:
            focal_length_x = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_x is None else
                focal_length_x,
                dtype=dtype)

        if focal_length_y is None or type(focal_length_y) == float:
            focal_length_y = torch.full(
                [batch_size],
                self.FOCAL_LENGTH if focal_length_y is None else
                focal_length_y,
                dtype=dtype)

        self.register_buffer('focal_length_x', focal_length_x)
        self.register_buffer('focal_length_y', focal_length_y)

        if center is None:
            center = torch.zeros([batch_size, 2], dtype=dtype)
        self.register_buffer('center', center)
        self.register_buffer('center_fix', center)

        if rotation is None:
            rotation = torch.eye(
                3, dtype=dtype).unsqueeze(dim=0).repeat(batch_size, 1, 1)

        rotation = nn.Parameter(rotation, requires_grad=False)
        self.register_parameter('rotation', rotation)

        if translation is None:
            translation = torch.zeros([batch_size, 3], dtype=dtype)

        translation = nn.Parameter(translation,
                                   requires_grad=True)
        self.register_parameter('translation', translation)
        
    

    def forward(self, points):
        device = points.device
        
        with torch.no_grad():
            camera_mat = torch.zeros([self.batch_size, 2, 2],
                                     dtype=self.dtype, device=points.device)
            camera_mat[:, 0, 0] = self.focal_length_x
            camera_mat[:, 1, 1] = self.focal_length_y

        camera_transform = transform_mat(self.rotation,
                                         self.translation.unsqueeze(dim=-1))
        homog_coord = torch.ones(list(points.shape)[:-1] + [1],
                                 dtype=points.dtype,
                                 device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk',
                                        [camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2],
                               projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) \
            + self.center.unsqueeze(dim=1)
        return img_points

    
class ClothRenderer(object):
    
    def __init__(self, resolution=512, focal_distance=1.6):
        self.device = torch.device("cuda:0")

        self.img_size = resolution
        self.render_size = resolution
        self.focal_distance = focal_distance
        self.renderer = self.__get_renderer(self.render_size, self.focal_distance)
        
        obj_filename = os.path.join('../xuchen0214/mesh_texture/mesh/weiyi.obj')
        verts, faces, aux = load_obj(
                    obj_filename,
                    device=self.device,
                    load_textures=True)
        self.faces = faces.verts_idx
        self.verts = verts
        self.aux = aux
        
        self.transform_rotation_fb = RotateAxisAngle(-180, 'Y').cuda()
        
        self.center = verts.mean(0)
        self.scale = max((verts - self.center).abs().max(0)[0])
        
        
        # 卫衣
        _keys = []
        for _ in aux.texture_images.keys():
            if "默认织物" in _:
                _keys.append(_)
        self.tex_lst = [aux.texture_images[i] for i in _keys]
        
        self.verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        tex_maps = aux.texture_images

        # Canonical Mesh(不考虑纹理)
        # tex_maps is a dictionary of {material name: texture image}.
        texture_image = self.tex_lst[0]
        texture_image = texture_image[None, ...].to(self.device)  # (1, H, W, 3)
        self.texture_image = texture_image
        self.texture = TexturesUV(maps=texture_image, faces_uvs=self.faces[None], verts_uvs=self.verts_uvs)
        self.canonical_mesh = Meshes([self.normalize_vertex(self.verts)], [self.faces], self.texture)
        
    
    def normalize_vertex(self, verts):
        # Normalizing
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        
        verts = verts - center
        verts = verts * (1/float(scale))
        
        return verts
    
    def render_image(self, texture_image, side='front'):
        texture = TexturesUV(maps=texture_image, faces_uvs=self.faces[None], verts_uvs=self.verts_uvs)
        
        # 0) normalizing vertex
        verts = self.normalize_vertex(self.verts)
        
        # pick front and back
        if side == 'front':
            mesh = Meshes([self.transform_rotation_fb.transform_points(verts)], [self.faces], texture)
        else:
            mesh = Meshes([verts], [self.faces], texture)
            
        target_images = self.renderer(mesh)
        
        out = (target_images[0].cpu().numpy()[:, :, :3] * 255.)

        cv2.imwrite("rendered_gaga_part.jpg", out.astype(np.uint8))

        return out

    def __get_renderer(self, render_size, focal_distance=2):
        
        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        
        R, T = look_at_view_transform(focal_distance, 180, 0)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.cameras = cameras
        
        raster_settings = RasterizationSettings(
            image_size=render_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        
        # (TODO) SoftPhong -> PBR(我自己写的).
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader = SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
            )
        )
        

        return renderer
    


if __name__ == "__main__":
    focal_length = 1.7
    cloth = ClothRenderer(resolution=512, focal_distance=focal_length)
    R, T = cloth.cameras.R.cuda(), cloth.cameras.T.cuda()
    cam = PerspectiveCamera(rotation=R, translation=T,
                            focal_length_x=focal_length, focal_length_y=focal_length).to(cloth.device)
    
    img = cloth.render_image(cloth.texture_image, side='front')

    # project normalized vertex to image space
    # verts_2d = cam(cloth.normalize_vertex(cloth.verts)[::10].unsqueeze(0))[0]
    # 2023.02.16 去除leo和reo对应的顶点!
    # verts_2d = cam(cloth.normalize_vertex(cloth.verts)[[156091, 155776, 164591, 159019, 71882, 74352, 132508, 145711, 152085, 154224, 152613, 148970]].unsqueeze(0))[0]
    verts_2d = cam(cloth.normalize_vertex(cloth.verts)[[156091, 155776, 164591, 159019, 71882, 74352, 152085, 154224, 152613, 148970]].unsqueeze(0))[0]

    # conversion from OpenGL coordinate to OpenCV coordinate
    verts_2d[:,] = -verts_2d[:,]

    # conversion from [-1,1] to [0,512]
    verts_2d = (verts_2d+1)/2*512

    for vert in verts_2d:
        cv2.circle(img, (int(vert[0]), int(vert[1])), 1, (0, 0, 0), -1)

    cv2.imwrite('cloth.png', img)
    
    # des_point_g = torch.tensor([-0.37184, -0.044689, 2.6069]).unsqueeze(0).unsqueeze(0)
    # des_point_l = torch.tensor([-0.37184, 2.6069, 0.044689]).unsqueeze(0).unsqueeze(0)
    # How to reprojection to resolution 512x512 image space???
    
    import pdb; pdb.set_trace()