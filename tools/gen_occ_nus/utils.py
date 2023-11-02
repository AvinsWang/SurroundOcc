import os
import torch
import numpy as np
import open3d as o3d
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyquaternion import Quaternion
from mmcv.ops.points_in_boxes import points_in_boxes_cpu, points_in_boxes_all
import chamfer


def pc_project(pc, M):
    """ pc with label, project to given coordinate

    Args:
        pc (np.array): Nx4
        M (np.array): 4x4 project matrix, e.g. or it's inv
            | R T |
            | 0 1 |
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


proj_M = get_3d_project_matrix


def get_lidar2cam(nusc, sample_tk, cam_name_lis):
    """ Get lidar to camera transform matrix

    Args:
        nusc (Nuscenes): Nuscenes dev-kit object
        sample_tk (str): Sample token
        cam_name_lis (list): [CAM_FRONT, ...], which camere to calc

    Returns:
        dict: _description_
    """
    my_sample = nusc.get('sample', sample_tk)
    lidar_tk = my_sample['data']['LIDAR_TOP']

    res_dic = dict()
    for cam_name in cam_name_lis:
        cam_tk = my_sample['data'][cam_name]
        lidar_dic = nusc.get('sample_data', lidar_tk)
        cam_dic = nusc.get('sample_data', cam_tk)

        img_ego = nusc.get('ego_pose', cam_dic['ego_pose_token'])
        img_cali = nusc.get('calibrated_sensor', cam_dic['calibrated_sensor_token'])

        lidar_ego = nusc.get('ego_pose', lidar_dic['ego_pose_token'])
        lidar_cali = nusc.get('calibrated_sensor', lidar_dic['calibrated_sensor_token'])

        cam2img = get_intrinsic_matrix(img_cali['camera_intrinsic'])
        cam2cam_ego = proj_M(img_cali['rotation'], img_cali['translation'])
        cam_ego2global = proj_M(img_ego['rotation'], img_ego['translation'])
        lidar2lidar_ego = proj_M(lidar_cali['rotation'], lidar_cali['translation'])
        lidar_ego2global = proj_M(lidar_ego['rotation'], lidar_ego['translation'])

        lidar2cam = inv(cam2cam_ego) @ inv(cam_ego2global) @ lidar_ego2global @ lidar2lidar_ego

        res_dic[cam_name] = dict(
            img_tk=cam_tk,
            img_name=cam_dic['filename'],
            cam2cam_ego=cam2cam_ego,
            cam_ego2global=cam_ego2global,
            lidar_ego2global=lidar_ego2global,
            lidar2lidar_ego=lidar2lidar_ego,
            intrinsic=cam2img,
            lidar2cam=lidar2cam,
        )
    return res_dic


def view_points(points, intrinsic, normalize) -> np.ndarray:
    """ Camera coordinate to Image coordinate

    Args:
        points (np.ndarray): N x 4
        intrinsic (np.ndarray): 4 x 4 intrinsic matrix
        normalize (bool): d(u, v, 1) -> (u, v, 1)

    Returns:
        np.ndarray: Nx4
    """

    # Do operation in homogenous coordinates.
    points = pc_project(points, intrinsic)

    if normalize:
        points[:, :3] = points[:, :3] / points[:, 2].reshape(-1, 1)

    return points


def proj_pc2img(pc_pts, lidar2cam, intrinsic, H, W):
    """ Project lidar point cloud to image plane

    Args:
        pc_pts (np.array): Nx4 point cloud on lidar coordinate
        lidar2cam (np.array): 4x4 transform matrix
        intrinsic (np.array): 4x4 intrinsic matrix
        H (int): Image H
        W (int): Image W

    Returns:
        np.array: Nx4, transformed points in image range
    """
    img_pts = pc_project(pc_pts, lidar2cam)

    depths = img_pts[:, 2]

    img_pts = view_points(img_pts, intrinsic, normalize=True)

    mask = ((depths > 1.0)
            & (img_pts[:, 0] > 1)
            & (img_pts[:, 0] < W - 1)
            & (img_pts[:, 1] > 1)
            & (img_pts[:, 1] < H - 1))
    img_pts = img_pts[mask]

    return img_pts


def fill_empty_holes(pc, is_calc_normals=False, max_nn=30,
                     depth=10, n_threads=-1, min_density=0.1):
    """ Fill empty holes with mesh reconstruction

    Args:
        pc (np.array): Nx4|Nx3
        is_calc_normals (bool, optional): _description_. Defaults to False.
        max_nn (int, optional): _description_. Defaults to 30.
        depth (int, optional): _description_. Defaults to 10.
        n_threads (int, optional): _description_. Defaults to -1.
        min_density (float, optional): _description_. Defaults to 0.1.

    Returns:
        np.array: Nx3
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    if is_calc_normals:
        _param = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        pcd.estimate_normals(search_param=_param)
        pcd.orient_normals_towards_camera_location()    # ?

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth, n_threads=n_threads)

    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    return np.asarray(mesh.vertices, dtype=float)


def get_grid_index(shape):
    """ Get grid array, grid[i][j][k]=val"""
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])
    z = np.linspace(0, shape[2] - 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    return vv


def get_dense_seg_label(dense_pc, seg_pc, device='cuda:0'):
    """ Assign label to dense_pc from seg_pc

    Args:
        dense_pc (_type_): Nx3
        seg_pc (_type_): Nx4

    Returns:
        np.array: Nx4
    """
    x = torch.from_numpy(dense_pc).float()[None, ...]
    y = torch.from_numpy(seg_pc[:, :3]).float()[None, ...]
    x = x.to(device)
    y = y.to(device)
    _, _, idx, _ = chamfer.forward(x, y)
    idx = idx[0].cpu().detach().numpy().astype(int)

    dense_seg_pc = seg_pc[:, 3][idx]
    res = np.concatenate([dense_pc, dense_seg_pc[..., None]], axis=1)
    torch.cuda.empty_cache()
    return res


def assign_empty_pcd(pcd, cls_id_lis, device='cuda:0'):
    """ Assign specify class id with nearset class

    Args:
        pcd (np.array): Nx4
        cls_id (list): [cls_id, ...]

    Returns:
        np.array: Nx4
    """

    mask_empty = np.ones(len(pcd))
    for cls_id in cls_id_lis:
        mask_empty = np.logical_and(mask_empty, pcd[:, 3] == cls_id)

    if mask_empty.sum() > 0 and mask_empty.sum() < len(pcd):
        empty_pcd = pcd[mask_empty]
        seg_pcd = pcd[~mask_empty]
        labeled_pcd = get_dense_seg_label(empty_pcd[:, :3], seg_pcd, device)
        pcd = np.concatenate([labeled_pcd, seg_pcd], axis=0)
    return pcd


def voxelize(pc, pc_range, voxel_shape, voxel_size, mode='unitary'):
    """ Voxelize point cloud

    Args:
        pc (np.array): Nx4
        pc_range (np.array): point cloud range
        voxel_shape (list|np.array): Shape of voxel, x,y,z e.g.[200, 200, 16]
        voxel_size (np.array): Resolution on x,y,z, e.g. [0.5, 0.5, 0.5]
        mode (str, optional): Default as 'unitary'
            'unitary': Random choose a point in voxel, it meams if there are
                       multi-class, the final class of voxel is Random. It
                       should be used when only one class.
            'multi': Assign the category of most points as the voxel category

    Returns:
        np.array: voxel with voxel_shape, voxel[i][j][k]=cls_id
    """
    voxel_pc = pc.copy()
    voxel_pc = (voxel_pc - pc_range[:3]) / voxel_size
    voxel_pc = np.floor(voxel_pc).astype(int)
    voxel = np.zeros(voxel_shape)
    voxel[voxel_pc[:, 0], voxel_pc[:, 1], voxel_pc[:, 2]] = 1

    grid = get_grid_index(voxel.shape)
    fov_voxels = grid[voxel > 0]
    return fov_voxels


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


def pts_in_bbox(pc, bbox, device='cuda:0'):
    # pc: [_, 4], bbox: [_, 7], return [B, N_pt, ]
    _pc = torch.from_numpy(pc[:, :3][None, ...]).to(device).float()
    _bbox = torch.from_numpy(bbox[None, ...]).to(device).float()
    _res = points_in_boxes_all(_pc, _bbox).to('cpu')

    # cpu
    # points_in_boxes_cpu(torch.from_numpy(pc[:, :3][None, ...]),
    #                            torch.from_numpy(bbox[None, ...]))

    return _res


class NeighbourIndex:
    """ Get neighbour point index

    z         y
    |       .
    |    .
    | .
    0---------x

    Returns:
        _type_: _description_
    """
    @classmethod
    def add_self_point(cls, index):
        self_pc = np.array([0, 0, 0])[None]
        return np.concatenate([index, self_pc], axis=0)

    @classmethod
    def get_index_with_range(cls, x_r, y_r, z_r):
        _index = list()
        for i in x_r:
            for j in y_r:
                for k in z_r:
                    if [i, j, k] == [0, 0, 0]:
                        continue
                    _index.append([i, j, k])
        index = np.array(_index, dtype=int)
        return index

    @classmethod
    def get_index(cls, mode='xyz8', is_add_self=False):
        """ Get different num neighbour point index

        Args:
            mode (str, optional): _description_. Defaults to 'xyz8'
                xyz8 : 8 corner points of the cub
                xy8  : 8 points arounded on xy plane
                xyz6 : 6 points of each surface
                xyz26: 26 points arounded of center

            is_add_self (bool, optional): add self [0,0,0] to index.

        Returns:
            np.array: Nx3
        """
        x_r, y_r, z_r = None, None, None
        if mode == 'xyz8':
            # for open3d render
            x_r = [-1, 1]
            y_r = [-1, 1]
            z_r = [-1, 1]
        elif mode == 'xy8':
            x_r = [0, -1, 1]
            y_r = [0, -1, 1]
            z_r = [0]
        elif mode == 'xyz6':
            index = np.stack([[1, -1, 0, 0, 0, 0],
                              [0, 0, 1, -1, 0, 0],
                              [0, 0, 0, 0, 1, -1]], axis=1)
        elif mode == 'xyz26':
            x_r = [0, -1, 1]
            y_r = [0, -1, 1]
            z_r = [0, -1, 1]
        if all([x_r, y_r, z_r]):
            index = cls.get_index_with_range(x_r, y_r, z_r)
        if is_add_self:
            index = cls.add_self_point(index)
        return index


def voxel_MA(pc, grid_size, mode='xy8'):
    """ Moving Average point cloud label

    Args:
        pc (_type_): _description_
        grid_size (_type_): _description_
        mode (str, optional): _description_. Defaults to 'xy8'.

    Returns:
        _type_: _description_
    """
    vertice = NeighbourIndex.get_index(mode=mode, is_add_self=True)
    nbr = len(vertice)
    pc_num = len(pc)

    vox = np.ones(grid_size) * -1
    vox[pc[:, 0], pc[:, 1], pc[:, 2]] = pc[:, 3]

    vertice = np.tile(vertice[None, ...], (len(pc), 1, 1))
    _pc = vertice + pc[:, :3][:, None, :]
    _pc = _pc.reshape(-1, 3)

    _pc[(_pc[:, 0] == -1), 0] = 0
    _pc[(_pc[:, 1] == -1), 1] = 0
    _pc[(_pc[:, 2] == -1), 2] = 0
    _pc[(_pc[:, 0] == grid_size[0]), 0] = grid_size[0] - 1
    _pc[(_pc[:, 1] == grid_size[1]), 1] = grid_size[1] - 1
    _pc[(_pc[:, 2] == grid_size[2]), 2] = grid_size[2] - 1

    label = vox[_pc[:, 0], _pc[:, 1], _pc[:, 2]]
    label_max = int(label.max())
    label = label.reshape(-1, nbr).astype(int)

    _pc = _pc.reshape(-1, nbr, 3)

    # TODO: batch化
    cnt_arr = np.zeros((pc_num, label_max + 1), dtype=int)
    for i, row in enumerate(label):
        cnt = np.bincount(row[row != -1])
        cnt_arr[i, 0:len(cnt)] = cnt

    new_label = cnt_arr.argmax(1)
    pc[:, 3] = new_label

    return pc


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
