import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import thinplate as tps

from numpy.testing import assert_allclose

def test_pytorch_grid():

    c_dst = np.array([
        [0., 0],
        [1., 0],    
        [1, 1],
        [0, 1],  
    ], dtype=np.float32)


    c_src = np.array([
        [10., 10],
        [20., 10],    
        [20, 20],
        [10, 20],  
    ], dtype=np.float32) / 40.

    theta = tps.tps_theta_from_points(c_src, c_dst)
    theta_r = tps.tps_theta_from_points(c_src, c_dst, reduced=True)

    np_grid = tps.tps_grid(theta, c_dst, (20,20))
    np_grid_r = tps.tps_grid(theta_r, c_dst, (20,20))
    
    pth_theta = torch.tensor(theta).unsqueeze(0)
    pth_grid = tps.torch.tps_grid(pth_theta, torch.tensor(c_dst), (1, 1, 20, 20)).squeeze().numpy()
    pth_grid = (pth_grid + 1) / 2 # convert [-1,1] range to [0,1]

    pth_theta_r = torch.tensor(theta_r).unsqueeze(0)
    pth_grid_r = tps.torch.tps_grid(pth_theta_r, torch.tensor(c_dst), (1, 1, 20, 20)).squeeze().numpy()
    pth_grid_r = (pth_grid_r + 1) / 2 # convert [-1,1] range to [0,1]

    assert_allclose(np_grid, pth_grid)
    assert_allclose(np_grid_r, pth_grid_r)
    assert_allclose(np_grid_r, np_grid)