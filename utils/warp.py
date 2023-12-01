# coding: UTF-8

import numpy as np
import numpy.linalg as npla
import cv2

def random_normal( size=(1,), trunc_val = 2.5, rnd_state=None ):
    if rnd_state is None:
        rnd_state = np.random
    len = np.array(size).prod()
    result = np.empty ( (len,) , dtype=np.float32)

    for i in range (len):
        while True:
            x = rnd_state.normal()
            if x >= -trunc_val and x <= trunc_val:
                break
        result[i] = (x / trunc_val)

    return result.reshape ( size )


def mls_rigid_deformation(vy, vx, src_pts, dst_pts, alpha=1.0, eps=1e-8):
    dst_pts = dst_pts[..., ::-1].astype(np.int16)
    src_pts = src_pts[..., ::-1].astype(np.int16)

    src_pts, dst_pts = dst_pts, src_pts

    grow = vx.shape[0]
    gcol = vx.shape[1]
    ctrls = src_pts.shape[0]

    reshaped_p = src_pts.reshape(ctrls, 2, 1, 1)
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))

    w = 1.0 / (np.sum((reshaped_p - reshaped_v).astype(np.float32) ** 2, axis=1) + eps) ** alpha
    w /= np.sum(w, axis=0, keepdims=True)

    pstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        pstar += w[i] * reshaped_p[i]

    vpstar = reshaped_v - pstar

    reshaped_mul_right = np.concatenate((vpstar[:,None,...],
                                         np.concatenate((vpstar[1:2,None,...],-vpstar[0:1,None,...]), 0)
                                         ), axis=1).transpose(2, 3, 0, 1)

    reshaped_q = dst_pts.reshape((ctrls, 2, 1, 1))
    
    qstar = np.zeros((2, grow, gcol), np.float32)
    for i in range(ctrls):
        qstar += w[i] * reshaped_q[i]   
        
    temp = np.zeros((grow, gcol, 2), np.float32)
    for i in range(ctrls):
        phat = reshaped_p[i] - pstar
        qhat = reshaped_q[i] - qstar

        temp += np.matmul(qhat.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1), 
                          
                          np.matmul( ( w[None, i:i+1,...] *
                                       np.concatenate((phat.reshape(1, 2, grow, gcol),
                                                       np.concatenate( (phat[None,1:2], -phat[None,0:1]), 1 )), 0)
                                      ).transpose(2, 3, 0, 1), reshaped_mul_right
                                   )
                         ).reshape(grow, gcol, 2)

    temp = temp.transpose(2, 0, 1)

    normed_temp = np.linalg.norm(temp, axis=0, keepdims=True)
    normed_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)
    nan_mask = normed_temp[0]==0

    transformers = np.true_divide(temp, normed_temp, out=np.zeros_like(temp), where= ~nan_mask) * normed_vpstar + qstar
    nan_mask_flat = np.flatnonzero(nan_mask)
    nan_mask_anti_flat = np.flatnonzero(~nan_mask)

    transformers[0][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[0][~nan_mask])
    transformers[1][nan_mask] = np.interp(nan_mask_flat, nan_mask_anti_flat, transformers[1][~nan_mask])

    return transformers    

def gen_pts(W, H, rnd_state=None):
    
    if rnd_state is None:
        rnd_state = np.random
        
    min_pts, max_pts = 4, 8
    n_pts = rnd_state.randint(min_pts, max_pts)
    
    min_radius_per = 0.00
    max_radius_per = 0.10
    pts = []
    
    for i in range(n_pts):
        while True:
            x, y = rnd_state.randint(W), rnd_state.randint(H)
            rad = min_radius_per + rnd_state.rand()*(max_radius_per-min_radius_per)
            
            intersect = False
            for px,py,prad,_,_ in pts:
                
                dist = npla.norm([x-px, y-py])
                if dist <= (rad+prad)*2:
                    intersect = True
                    break
            if intersect:
                continue   
            
            angle = rnd_state.rand()*(2*np.pi)
            x2 = int(x+np.cos(angle)*W*rad)
            y2 = int(y+np.sin(angle)*H*rad)
            
            break
        pts.append( (x,y,rad, x2,y2) )
        
    pts1 = np.array( [ [pt[0],pt[1]] for pt in pts ] )
    pts2 = np.array( [ [pt[-2],pt[-1]] for pt in pts ] )
    
    return pts1, pts2
    
    
def gen_warp_params(w, flip=False, rotation_range=[-10,10], scale_range=[-0.5, 0.5], tx_range=[-0.05, 0.05], ty_range=[-0.05, 0.05], rnd_state=None, warp_rnd_state=None):
    if rnd_state is None:
        rnd_state = np.random
    if warp_rnd_state is None:
        warp_rnd_state = np.random
    rw = None
    if w < 64:        
        rw = w
        w = 64
        
    rotation = rnd_state.uniform( rotation_range[0], rotation_range[1] )
    scale = rnd_state.uniform( 1/(1-scale_range[0]) , 1+scale_range[1] )
    tx = rnd_state.uniform( tx_range[0], tx_range[1] )
    ty = rnd_state.uniform( ty_range[0], ty_range[1] )
    p_flip = flip and rnd_state.randint(10) < 4

    #random warp V1
    cell_size = [ w // (2**i) for i in range(1,4) ] [ warp_rnd_state.randint(3) ]
    cell_count = w // cell_size + 1
    grid_points = np.linspace( 0, w, cell_count)
    mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
    mapy = mapx.T
    mapx[1:-1,1:-1] = mapx[1:-1,1:-1] + random_normal( size=(cell_count-2, cell_count-2), rnd_state=warp_rnd_state )*(cell_size*0.24)
    mapy[1:-1,1:-1] = mapy[1:-1,1:-1] + random_normal( size=(cell_count-2, cell_count-2), rnd_state=warp_rnd_state )*(cell_size*0.24)
    half_cell_size = cell_size // 2
    mapx = cv2.resize(mapx, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)
    mapy = cv2.resize(mapy, (w+cell_size,)*2 )[half_cell_size:-half_cell_size,half_cell_size:-half_cell_size].astype(np.float32)
    ##############
    
    # random warp V2
    # pts1, pts2 = gen_pts(w, w, rnd_state)
    # gridX = np.arange(w, dtype=np.int16)
    # gridY = np.arange(w, dtype=np.int16)
    # vy, vx = np.meshgrid(gridX, gridY)
    # drigid = mls_rigid_deformation(vy, vx, pts1, pts2)
    # mapy, mapx = drigid.astype(np.float32)
    ################
    
    #random transform
    random_transform_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, scale)
    random_transform_mat[:, 2] += (tx*w, ty*w)

    params = dict()
    params['mapx'] = mapx
    params['mapy'] = mapy
    params['rmat'] = random_transform_mat
    u_mat = random_transform_mat.copy()
    u_mat[:,2] /= w
    params['umat'] = u_mat
    params['w'] = w
    params['rw'] = rw
    params['flip'] = p_flip

    return params

def warp_by_params(params, img, can_warp, can_transform, can_flip, border_replicate, cv2_inter=cv2.INTER_CUBIC):
    rw = params['rw']
    
    if (can_warp or can_transform) and rw is not None:
        img = cv2.resize(img, (64,64), interpolation=cv2_inter)
        
    if can_warp:
        img = cv2.remap(img, params['mapx'], params['mapy'], cv2_inter )
    if can_transform:
        img = cv2.warpAffine( img, params['rmat'], (params['w'], params['w']), borderMode=(cv2.BORDER_REPLICATE if border_replicate else cv2.BORDER_CONSTANT), flags=cv2_inter )
    
    
    if (can_warp or can_transform) and rw is not None:
        img = cv2.resize(img, (rw,rw), interpolation=cv2_inter)
    
    if len(img.shape) == 2:
        img = img[...,None]
    if can_flip and params['flip']:
        img = img[:,::-1,...]
    return 