"""
Bilateral Normal Integration (BiNI)
"""
__author__ = "Xu Cao <cao.xu@ist.osaka-u.ac.jp>; Yuliang Xiu <yuliang.xiu@tue.mpg.de>"
__copyright__ = "Copyright (C) 2022 Xu Cao; Yuliang Xiu"
__version__ = "2.0"

import trimesh
import pyvista as pv
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg
from tqdm.auto import tqdm
import time

pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)

# Define helper functions for moving masks in different directions
def move_left(mask): return cp.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]  # Shift the input mask array to the left by 1, filling the right edge with zeros.
def move_right(mask): return cp.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]  # Shift the input mask array to the right by 1, filling the left edge with zeros.
def move_top(mask): return cp.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]  # Shift the input mask array up by 1, filling the bottom edge with zeros.
def move_bottom(mask): return cp.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]  # Shift the input mask array down by 1, filling the top edge with zeros.
def move_top_left(mask): return cp.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]  # Shift the input mask array up and to the left by 1, filling the bottom and right edges with zeros.
def move_top_right(mask): return cp.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]  # Shift the input mask array up and to the right by 1, filling the bottom and left edges with zeros.
def move_bottom_left(mask): return cp.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]  # Shift the input mask array down and to the left by 1, filling the top and right edges with zeros.
def move_bottom_right(mask): return cp.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  # Shift the input mask array down and to the right by 1, filling the top and left edges with zeros.


def depth_to_pcd(depth, K=None, scale=1.0):
    H, W, _ = depth.shape
    dtype = depth.dtype
    # gy, gx = cp.meshgrid(
    #     cp.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, dtype=dtype),
    #     cp.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, dtype=dtype),
    #     indexing='ij',
    # )
    gy, gx = cp.meshgrid(
        cp.arange(H, dtype=dtype),
        cp.arange(W, dtype=dtype),
        indexing='ij',
    )
    gx = cp.flip(gx, axis=0)
    if K is None:
        xyz = cp.stack([gx * scale, gy * scale, depth[:, :, 0]], axis=-1)
    else:
        xy1 = cp.stack([gx, gy, cp.ones_like(gx)], axis=-1)
        xyz = cp.matmul(xy1, cp.linalg.inv(cp.asarray(K)).T) * depth
    xyz = xyz - xyz.reshape(-1, 3).mean(axis=0, keepdims=True)
    # scale = max(H, W, depth.max() - depth.min()) / 2.0
    scale = max(H, W) / 2.0
    xyz = xyz / scale
    xyz[..., 1] = -xyz[..., 1]
    xyz[..., 2] = -xyz[..., 2]
    return xyz


def normal_to_dnz(normal, mask, K=None, scale=1.0):
    H, W, _ = normal.shape
    dtype = normal.dtype
    nx = normal[mask, 1]
    ny = normal[mask, 0]
    nz = - normal[mask, 2]
    if K is not None:
        gy, gx = cp.meshgrid(
            cp.arange(H, dtype=dtype),
            cp.arange(W, dtype=dtype),
            index='ij',
        )
        gx = cp.flip(gx, axis=0)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = gx[mask] - cx
        v = gy[mask] - cy
        nz_u = u * nx + v * ny + fx * nz
        nz_v = u * nx + v * ny + fy * nz
    else:
        nz_u = nz / scale
        nz_v = nz / scale
    return nx, ny, nz, nz_u, nz_v


def dnz_to_dxdy(nz_u, nz_v, mask):
    num_pixel = cp.sum(mask)

    # Generate an integer index array with the same shape as the mask.
    pixel_idx = cp.zeros_like(mask, dtype=int)
    # Assign a unique integer index to each True value in the mask.
    pixel_idx[mask] = cp.arange(num_pixel)

    # Create boolean masks representing the presence of neighboring pixels in each direction.
    has_left_mask = cp.logical_and(move_right(mask), mask)
    has_right_mask = cp.logical_and(move_left(mask), mask)
    has_bottom_mask = cp.logical_and(move_top(mask), mask)
    has_top_mask = cp.logical_and(move_bottom(mask), mask)

    # Extract the horizontal and vertical components of the normal vectors for the neighboring pixels.
    nz_left = nz_v[has_left_mask[mask]]
    nz_right = nz_v[has_right_mask[mask]]
    nz_top = nz_u[has_top_mask[mask]]
    nz_bottom = nz_u[has_bottom_mask[mask]]

    # Create sparse matrices representing the partial derivatives for each direction.
    # top/bottom/left/right = vertical positive/vertical negative/horizontal negative/horizontal positive
    # The matrices are constructed using the extracted normal components and pixel indices.
    data = cp.stack([-nz_left, nz_left], -1).flatten()
    indices = cp.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_right, nz_right], -1).flatten()
    indices = cp.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_top, nz_top], -1).flatten()
    indices = cp.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = cp.stack([-nz_bottom, nz_bottom], -1).flatten()
    indices = cp.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def generate_faces(H:int, W:int):
    ix = np.arange(H - 1, dtype=np.int64)
    iy = np.arange(W - 1, dtype=np.int64)
    ox = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    oy = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)
    faces = ((ix[:, None, None, None] + ox) % H) * W + ((iy[None, :, None, None] + oy) % W)
    return faces


def sigmoid(x, k=1):
    return 1 / (1 + cp.exp(-k * x))


def bilateral_normal_integration_function(
    normal_map,
    normal_mask,
    rgb=None,
    k=2,
    lambda1=0,
    depth_map=None,
    depth_mask=None,
    K=None,
    scale=1.0,
    max_iter=150,
    tol=1e-4,
    cg_max_iter=5000,
    cg_tol=1e-3,
):
    """
    This function performs the bilateral normal integration algorithm, as described in the paper.
    It takes as input the normal map, normal mask, and several optional parameters to control the integration process.

    :param normal_map: A normal map, which is an image where each pixel's color encodes the corresponding 3D surface normal.
    :param normal_mask: A binary mask that indicates the region of interest in the normal_map to be integrated.
    :param k: A parameter that controls the stiffness of the surface.
              The smaller the k value, the smoother the surface appears (fewer discontinuities).
              If set as 0, a smooth surface is obtained (No discontinuities), and the iteration should end at step 2 since the surface will not change with iterations.

    :param depth_map: (Optional) An initial depth map to guide the integration process.
    :param depth_mask: (Optional) A binary mask that indicates the valid depths in the depth_map.

    :param lambda1 (Optional): A regularization parameter that controls the influence of the depth_map on the final result.
                               Required when depth map is input.
                               The larger the lambda1 is, the result more close to the initial depth map (fine details from the normal map are less reflected)

    :param K: (Optional) A 3x3 camera intrinsic matrix, used for perspective camera models. If not provided, the algorithm assumes an orthographic camera model.
    :param scale: (Optional) The pixel size in the world coordinates. Default value is 1.
                                 Used only in the orthographic camera mdoel.
                                 Default value should be fine, unless you know the true value of the pixel size in the world coordinates.
                                 Do not adjust it in perspective camera model.

    :param max_iter: (Optional) The maximum number of iterations for the optimization process. Default value is 150.
                                If set as 1, a smooth surface is obtained (No discontinuities).
                                Default value should be fine.
    :param tol:  (Optional) The tolerance for the relative change in energy to determine the convergence of the optimization process. Default value is 1e-4.
                            The larger, the iteration stops faster, but the discontinuity preservation quality might be worse. (fewer discontinuities)
                            Default value should be fine.

    :param cg_max_iter: (Optional) The maximum number of iterations for the Conjugate Gradient solver. Default value is 5000.
                                   Default value should be fine.
    :param cg_tol: (Optional) The tolerance for the Conjugate Gradient solver. Default value is 1e-3.
                              Default value should be fine.

    :return: depth_map: The resulting depth map after the bilateral normal integration process.
             surface: A pyvista PolyData mesh representing the 3D surface reconstructed from the depth map.
             wu_map: A 2D image that represents the horizontal smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             wv_map: A 2D image that represents the vertical smoothness weight for each pixel. (green for smooth, blue/red for discontinuities)
             energy_list: A list of energy values during the optimization process.
    """
    # To avoid confusion, we list the coordinate systems in this code as follows
    #
    # pixel coordinates         camera coordinates     normal coordinates (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                       (o is the optical center;
    #                        xy-plane is parallel to the image plane;
    #                        +z is the viewing direction.)
    #
    # The input normal map should be defined in the normal coordinates.
    # The camera matrix K should be defined in the camera coordinates.
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]
    # I forgot why I chose the awkward coordinate system after getting used to opencv convention :(
    # but I won't touch the working code.

    normal_map = cp.asarray(normal_map)
    normal_mask = cp.asarray(normal_mask)
    if depth_map is not None:
        depth_map = cp.asarray(depth_map)
        depth_mask = cp.asarray(depth_mask)
    num_normals = cp.sum(normal_mask).item()
    # transfer the normal map from the normal coordinates to the camera coordinates
    nx, ny, nz, nz_u, nz_v = normal_to_dnz(normal_map, normal_mask, K=K, scale=scale)
    # right, left, top, bottom
    A3, A4, A1, A2 = dnz_to_dxdy(nz_u, nz_v, normal_mask)

    pixel_idx = cp.zeros_like(normal_mask, dtype=int)
    pixel_idx[normal_mask] = cp.arange(num_normals)
    pixel_idx_flat = cp.arange(num_normals)
    pixel_idx_flat_indptr = cp.arange(num_normals + 1)

    has_left_mask = cp.logical_and(move_right(normal_mask), normal_mask)
    has_left_mask_left = move_left(has_left_mask)
    has_right_mask = cp.logical_and(move_left(normal_mask), normal_mask)
    has_right_mask_right = move_right(has_right_mask)
    has_bottom_mask = cp.logical_and(move_top(normal_mask), normal_mask)
    has_bottom_mask_bottom = move_bottom(has_bottom_mask)
    has_top_mask = cp.logical_and(move_bottom(normal_mask), normal_mask)
    has_top_mask_top = move_top(has_top_mask)

    has_left_mask_flat = has_left_mask[normal_mask]
    has_right_mask_flat = has_right_mask[normal_mask]
    has_bottom_mask_flat = has_bottom_mask[normal_mask]
    has_top_mask_flat = has_top_mask[normal_mask]

    has_left_mask_left_flat = has_left_mask_left[normal_mask]
    has_right_mask_right_flat = has_right_mask_right[normal_mask]
    has_bottom_mask_bottom_flat = has_bottom_mask_bottom[normal_mask]
    has_top_mask_top_flat = has_top_mask_top[normal_mask]

    nz_left_square = nz_v[has_left_mask_flat] ** 2
    nz_right_square = nz_v[has_right_mask_flat] ** 2
    nz_top_square = nz_u[has_top_mask_flat] ** 2
    nz_bottom_square = nz_u[has_bottom_mask_flat] ** 2

    pixel_idx_left_center = pixel_idx[has_left_mask]
    pixel_idx_right_right = pixel_idx[has_right_mask_right]
    pixel_idx_top_center = pixel_idx[has_top_mask]
    pixel_idx_bottom_bottom = pixel_idx[has_bottom_mask_bottom]

    pixel_idx_left_left_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_left_mask_left_flat)])
    pixel_idx_right_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_right_mask_flat)])
    pixel_idx_top_top_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_top_mask_top_flat)])
    pixel_idx_bottom_center_indptr = cp.concatenate([cp.array([0]), cp.cumsum(has_bottom_mask_flat)])

    # initialization
    if depth_map is not None:
        depth_mask_flat = depth_mask[normal_mask].astype(bool)  # shape: (num_normals,)
        z_prior = cp.log(depth_map)[normal_mask] if K is not None else depth_map[normal_mask]  # shape: (num_normals,)
        z_prior[~depth_mask_flat] = 0
    wu = 0.5 * cp.ones(num_normals, float)
    wv = 0.5 * cp.ones(num_normals, float)
    z = cp.zeros(num_normals, float)
    energy = cp.sum(wu * (A1.dot(z) + nx) ** 2) + \
             cp.sum((1 - wu) * (A2.dot(z) + nx) ** 2) + \
             cp.sum(wv * (A3.dot(z) + ny) ** 2) + \
             cp.sum((1 - wv) * (A4.dot(z) + ny) ** 2)
    energy_list = []

    tic = time.time()
    pbar = tqdm(range(max_iter))
    for i in pbar:
        ################################################################################################################
        # I am manually computing A_mat = A.T @ W @ A here. It saves 2/3 time compared to the simpliest way A.T @ W @ A.
        # A.T @ W @ A can take more time than you think when the normal map become larger.
        # The diaganol matrix W=diag([wu, 1-wu, wv, 1-wv]) needs not be explicited defined in this case.
        #
        data_term_top = wu[has_top_mask_flat] * nz_top_square
        data_term_bottom = (1 - wu[has_bottom_mask_flat]) * nz_bottom_square
        data_term_left = (1 - wv[has_left_mask_flat]) * nz_left_square
        data_term_right = wv[has_right_mask_flat] * nz_right_square

        diagonal_data_term = cp.zeros(num_normals)
        diagonal_data_term[has_left_mask_flat] += data_term_left
        diagonal_data_term[has_left_mask_left_flat] += data_term_left
        diagonal_data_term[has_right_mask_flat] += data_term_right
        diagonal_data_term[has_right_mask_right_flat] += data_term_right
        diagonal_data_term[has_top_mask_flat] += data_term_top
        diagonal_data_term[has_top_mask_top_flat] += data_term_top
        diagonal_data_term[has_bottom_mask_flat] += data_term_bottom
        diagonal_data_term[has_bottom_mask_bottom_flat] += data_term_bottom
        if depth_map is not None:
            diagonal_data_term[depth_mask_flat] += lambda1

        A_mat_d = csr_matrix((diagonal_data_term, pixel_idx_flat, pixel_idx_flat_indptr),
                             shape=(num_normals, num_normals))

        A_mat_left_odu = csr_matrix((-data_term_left, pixel_idx_left_center, pixel_idx_left_left_indptr),
                                    shape=(num_normals, num_normals))
        A_mat_right_odu = csr_matrix((-data_term_right, pixel_idx_right_right, pixel_idx_right_center_indptr),
                                     shape=(num_normals, num_normals))
        A_mat_top_odu = csr_matrix((-data_term_top, pixel_idx_top_center, pixel_idx_top_top_indptr),
                                   shape=(num_normals, num_normals))
        A_mat_bottom_odu = csr_matrix((-data_term_bottom, pixel_idx_bottom_bottom, pixel_idx_bottom_center_indptr),
                                      shape=(num_normals, num_normals))

        A_mat_odu = A_mat_top_odu + A_mat_bottom_odu + A_mat_right_odu + A_mat_left_odu
        A_mat = A_mat_d + A_mat_odu + A_mat_odu.T  # diagnol + upper triangle + lower triangle matrix
        ################################################################################################################

        D = csr_matrix((1 / cp.clip(diagonal_data_term, 1e-5, None), pixel_idx_flat, pixel_idx_flat_indptr),
                       shape=(num_normals, num_normals))  # Jacobi preconditioner.
        b_vec = A1.T @ (wu * (-nx)) \
                + A2.T @ ((1 - wu) * (-nx)) \
                + A3.T @ (wv * (-ny)) \
                + A4.T @ ((1 - wv) * (-ny))

        if depth_map is not None:
            b_vec += lambda1 * z_prior
            offset = cp.mean((z_prior - z)[depth_mask_flat])
            z = z + offset

        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, tol=cg_tol)
        del A_mat, b_vec, wu, wv

        # Update weights
        wu = sigmoid((A2.dot(z)) ** 2 - (A1.dot(z)) ** 2, k)  # top
        wv = sigmoid((A4.dot(z)) ** 2 - (A3.dot(z)) ** 2, k)  # right

        # Check for convergence
        energy_old = energy
        energy = cp.sum(wu * (A1.dot(z) + nx) ** 2) + \
                 cp.sum((1 - wu) * (A2.dot(z) + nx) ** 2) + \
                 cp.sum(wv * (A3.dot(z) + ny) ** 2) + \
                 cp.sum((1 - wv) * (A4.dot(z) + ny) ** 2)

        energy_list.append(energy)
        relative_energy = cp.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3e}"
            f" relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    del A1, A2, A3, A4, nx, ny
    toc = time.time()
    print(f"Total time: {toc - tic:.3f} sec")

    depth_map = cp.full_like(normal_mask, fill_value=np.nan, dtype=cp.float32)
    depth_map[normal_mask] = z
    depth_map = cp.nan_to_num(depth_map, copy=False, nan=z.max())
    if K is not None:
        depth_map = cp.exp(depth_map)
    vertices = cp.asnumpy(depth_to_pcd(depth_map[:, :, None], K=K, scale=scale)).reshape(-1, 3)
    vertex_colors = rgb.reshape(-1, 3) if rgb is not None else None
    vertex_normals = cp.asnumpy(normal_map).reshape(-1, 3)
    H, W, _ = normal_map.shape
    faces = np.flip(generate_faces(H, W), -1).reshape(-1, 3)
    surface = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
        process=False,
    )

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = cp.ones_like(normal_mask) * cp.nan
    wu_map[normal_mask] = wv

    wv_map = cp.ones_like(normal_mask) * cp.nan
    wv_map[normal_mask] = wu

    depth_map = cp.asnumpy(depth_map)
    wu_map = cp.asnumpy(wu_map)
    wv_map = cp.asnumpy(wv_map)

    return depth_map, surface, wu_map, wv_map, energy_list


if __name__ == '__main__':
    import cv2
    import argparse
    import os
    import warnings
    warnings.filterwarnings('ignore')
    # To ignore the possible overflow runtime warning: overflow encountered in exp return 1 / (1 + cp.exp(-k * x)).
    # This overflow issue does not affect our results as cp.exp will correctly return 0.0 when -k * x is massive.

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise FileNotFoundError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=dir_path)
    parser.add_argument('-k', type=float, default=2)
    parser.add_argument('-i', '--iter', type=int, default=150)
    parser.add_argument('-t', '--tol', type=float, default=1e-4)
    parser.add_argument('--cgiter', type=int, default=5000)
    parser.add_argument('--cgtol', type=float, default=1e-3)
    arg = parser.parse_args()

    normal_map = cv2.cvtColor(cv2.imread(os.path.join(
        arg.path, "normal_map.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    if normal_map.dtype is np.dtype(np.uint16):
        normal_map = normal_map/65535 * 2 - 1
    else:
        normal_map = normal_map/255 * 2 - 1

    try:
        mask = cv2.imread(os.path.join(arg.path, "mask.png"), cv2.IMREAD_GRAYSCALE).astype(bool)
    except:
        mask = np.ones(normal_map.shape[:2], bool)

    if os.path.exists(os.path.join(arg.path, "K.txt")):
        K = np.loadtxt(os.path.join(arg.path, "K.txt"))
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=K,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol,
                                                                                       cg_max_iter=arg.cgiter,
                                                                                       cg_tol=arg.cgtol)
    else:
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(normal_map=normal_map,
                                                                                       normal_mask=mask,
                                                                                       k=arg.k,
                                                                                       K=None,
                                                                                       max_iter=arg.iter,
                                                                                       tol=arg.tol,
                                                                                       cg_max_iter=arg.cgiter,
                                                                                       cg_tol=arg.cgtol)

    # save the resultant polygon mesh and discontinuity maps.
    cp.save(os.path.join(arg.path, "energy"), cp.array(energy_list))
    surface.save(os.path.join(arg.path, f"mesh_k_{arg.k}.ply"), binary=False)
    wu_map = cv2.applyColorMap(
        (255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
    wv_map = cv2.applyColorMap(
        (255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
    wu_map[~mask] = 255
    wv_map[~mask] = 255
    cv2.imwrite(os.path.join(arg.path, f"wu_k_{arg.k}.png"), wu_map)
    cv2.imwrite(os.path.join(arg.path, f"wv_k_{arg.k}.png"), wv_map)
    print(f"saved {arg.path}")
