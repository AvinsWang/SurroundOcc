import torch
import numpy as np
import open3d as o3d
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import points_in_boxes_cpu
import chamfer


def pc_project(pc, M):
    """ pc with label, project to given coordinate

    Args:
        pc (np.array): Nx4
        M (np.array): 4x4 project matrix
    """
    labels = pc[:, 3]
    _pc = pc.copy()
    _pc[:, 3] = 1
    _pc = _pc @ M.T
    _pc[:, 3] = labels
    return _pc


def get_3d_project_matrix(rot_q, trans, size=4):
    """ Get 4x4 or 3x3 project matrix

    Args:
        rot_q (np.array): Quaternion
        trans (np.array): [x, y, z]
        size (int): 4x4 or 3x3 matrix

    Returns:
        M (np.array): project matrix
    """
    M = np.eye(4)
    rot_m = Quaternion(rot_q).rotation_matrix   # 3x3
    M[:3, :3] = rot_m
    M[:3, 3] = trans
    return M


def get_intrinsic_matrix(intrinsic, size=4):
    """ Get 4x4 or 3x3 camera intrinsice matrix

    Args:
        intrinsic (np.array): 3x3 matrix
        size (int, optional): matrix size. Defaults to 4.
    """
    if size == 4:
        M = np.eye(4)
        M[:3, :3] = intrinsic
    else:
        M = intrinsic
    return M


def fill_empty_holes(pc, is_calc_normals=False, radius=0.1, max_nn=30,
                     depth=10, n_threads=-1, min_density=0.1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    if is_calc_normals:
        _param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        pcd.estimate_normals(search_param=_param)
        pcd.orient_normals_towards_camera_location()    # 作用?

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth, n_threads=n_threads)

    if min_density:
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, min_density))
    return np.asarray(mesh.vertices, dtype=float)


def get_grid_index(shape):
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    return vv


def get_dense_seg_label(dense_pc, seg_pc):
    x = torch.from_numpy(dense_pc).cuda().float()[None, ...]
    y = torch.from_numpy(seg_pc[:, :3]).cuda().float()[None, ...]
    _, _, idx, _ = chamfer.forward(x, y)
    idx = idx[0].cpu().numpy().astype(int)

    dense_seg_pc = seg_pc[:, 3][idx]
    res = np.concatenate([dense_pc, dense_seg_pc[..., None]], axis=1)
    return res


def get_range_mask(pc, range_=None, is_mask=True):
    # range: [x_min, y_min, z_min, x_max, y_max, z_max]
    mask = ((pc[:, 0] > range_[0])
            & (pc[:, 0] < range_[3])
            & (pc[:, 1] > range_[1])
            & (pc[:, 1] < range_[4])
            & (pc[:, 2] > range_[2])
            & (pc[:, 2] < range_[5]))
    if is_mask:
        pc = pc[mask]
    return pc, mask


def pts_in_bbox(pc, bbox):
    # pc: [_, 4], bbox: [_, 7], return [B, N_pt, ]
    return points_in_boxes_cpu(torch.from_numpy(pc[:, :3][None, ...]),
                               torch.from_numpy(bbox[None, ...]))


# == deubg ==


def wlh2pts(lis):
    lis = np.array(lis)
    h_wlh = lis[3:6] / 2.
    lt = lis[:3] - h_wlh
    rb = lis[:3] + h_wlh
    return [lt, rb]


def show_bbox_on_pc(pc, boxes):
    fig, ax = plt.subplots()
    ax.scatter(pc[:, 0], pc[:, 1], s=.1)
    for box in boxes:
        wlh = box.wlh
        rad = box.orientation.yaw_pitch_roll[0]
        deg = np.rad2deg(rad) + 90
        rect = patches.Rectangle(box.center[:2], wlh[0], wlh[1], angle=deg,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
