
import numpy as np
from numpy.testing import assert_allclose
import thinplate as tps
    
def test_numpy_fit():
    c = np.array([
        [0., 0, 0.0],
        [1., 0, 0.0],
        [1., 1, 0.0],
        [0, 1, 0.0],
    ])

    theta = tps.TPS.fit(c)
    assert_allclose(theta, 0)
    assert_allclose(tps.TPS.z(c, c, theta), c[:, 2])

    c = np.array([
        [0., 0, 1.0],
        [1., 0, 1.0],
        [1., 1, 1.0],
        [0, 1, 1.0],
    ])

    theta = tps.TPS.fit(c)
    assert_allclose(theta[:-3], 0)
    assert_allclose(theta[-3:], [1, 0, 0])
    assert_allclose(tps.TPS.z(c, c, theta), c[:, 2], atol=1e-3)

    # reduced form
    theta = tps.TPS.fit(c, reduced=True)
    assert len(theta) == c.shape[0] + 2
    assert_allclose(theta[:-3], 0)
    assert_allclose(theta[-3:], [1, 0, 0])
    assert_allclose(tps.TPS.z(c, c, theta), c[:, 2], atol=1e-3)

    c = np.array([
        [0., 0, -.5],
        [1., 0, 0.5],
        [1., 1, 0.2],
        [0, 1, 0.8],
    ])

    theta = tps.TPS.fit(c)
    assert_allclose(tps.TPS.z(c, c, theta), c[:, 2], atol=1e-3)
    
def test_numpy_densegrid():

    # enlarges a small rectangle to full view

    import cv2

    img = np.zeros((40, 40), dtype=np.uint8)
    img[10:21, 10:21] = 255

    c_dst = np.array([
        [0., 0],
        [1., 0],    
        [1, 1],
        [0, 1],  
    ])


    c_src = np.array([
        [10., 10],
        [20., 10],    
        [20, 20],
        [10, 20],  
    ]) / 40.

    theta = tps.tps_theta_from_points(c_src, c_dst)
    theta_r = tps.tps_theta_from_points(c_src, c_dst, reduced=True)

    grid = tps.tps_grid(theta, c_dst, (20,20))
    grid_r = tps.tps_grid(theta_r, c_dst, (20,20))

    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    warped = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

    assert img.min() == 0.
    assert img.max() == 255.
    assert warped.shape == (20,20)
    assert warped.min() == 255.
    assert warped.max() == 255.
    assert np.linalg.norm(grid.reshape(-1,2) - grid_r.reshape(-1,2)) < 1e-3
