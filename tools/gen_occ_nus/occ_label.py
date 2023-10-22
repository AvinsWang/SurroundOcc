import os
import torch
import numpy as np
import os.path as osp
from loguru import logger
from tools.gen_occ_nus import utils
from scipy.spatial.transform import Rotation


pj_M = utils.get_3d_project_matrix
pj_K = utils.get_intrinsic_matrix
pc_project = utils.pc_project


class OccLabel:
    def __init__(self, nusc, cfg, idx=None) -> None:
        self.nusc = nusc
        self.idx = idx
        self.cfg = cfg
        # # converted frame count
        self.frame_cnt = 0

    def load_val_lis(self):
        _val = list()
        with open(self.nusc_val_fpath, 'r') as file:
            for item in file:
                _val.append(item[:-1])
        self.cfg.sampe_val_tk_lis = _val

    def get_all_lidar_of_scene(self, scene):
        cur_sample_tk = scene['first_sample_token']
        _sample = self.nusc.get('sample', cur_sample_tk)
        cur_tk = _sample['data']['LIDAR_TOP']

        sensor_lis = list()
        while cur_tk != '':
            sensor_lis.append(cur_tk)
            dic = self.nusc.get('sample_data', cur_tk)
            cur_tk = dic['next']
        return sensor_lis

    def load_lidar_seg(self, token):
        path = osp.join(self.cfg['data_root'],
                        self.nusc.get('lidarseg', token)['filename'])
        return np.fromfile(path, dtype=np.uint8)

    def get_sensor2glb(self, lidar_or_cam_tk=None, dic=None, is_cam=False):
        if lidar_or_cam_tk is not None:
            dic = self.nusc.get('sample_data', lidar_or_cam_tk)
        cali_dic = self.nusc.get('calibrated_sensor', dic['calibrated_sensor_token'])
        ego_dic = self.nusc.get('ego_pose', dic['ego_pose_token'])
        s2s_ego = pj_M(cali_dic['rotation'], cali_dic['translation'])
        s_ego2glb = pj_M(ego_dic['rotation'], ego_dic['translation'])
        s2glb = s_ego2glb @ s2s_ego
        out_dic = dict(
            sens2glb=s2glb,
            sens2sens_ego=s2s_ego,
            sens_ego2glb=s_ego2glb,
        )
        # notice: lidar_dic also have 'camera_intrinsic' but result is []
        if is_cam and 'camera_intrinsic' in cali_dic:
            out_dic.update(dict(
                intrinsic=pj_K(cali_dic['camera_intrinsic'])
            ))
        return out_dic

    def get_bboxes(self, boxes):
        obj_tk_lis, obj_cate_lis, obj_cate_id_lis = list(), list(), list()
        gt_bbox_3d = list()
        for box in boxes:
            _anno = self.nusc.get('sample_annotation', box.token)
            obj_tk_lis.append(_anno['instance_token'])
            obj_cate_lis.append(_anno['category_name'])
            # nusc cate_name -> nusc cate_id -> occ cate_id
            obj_cate_id_lis.append(self.cfg.labels.nusc_id2occ_id[
                self.cfg.labels.nusc_name2id[_anno['category_name']]])

            # z - h/2, 0
            box.center[2] -= box.wlh[2] / 2
            box.center[2] -= 0.1
            box.wlh *= 1.1
            # 猜测, yaw 0 度为x轴方向, +90将 0 度转为y轴方向
            yaw = box.orientation.yaw_pitch_roll[0][None, ...] + np.pi / 2
            gt_bbox_3d.append(np.concatenate([box.center, box.wlh, yaw]))

        gt_bbox_3d = np.stack(gt_bbox_3d, axis=0).astype(np.float32)
        return gt_bbox_3d, obj_tk_lis, obj_cate_lis, obj_cate_id_lis

    def split_single_lidar_frame(self, lidar_tk, glb2lidar_ref_M):
        """ Split static scene and moving object in sigle one lidar frame

        Args:
            lidar_tk (str): nusc lidar token.
            glb2lidar_ref_M (np.array): 4x4 rotate matrix for converting lidar
                pcd to reference coordinate, e.g. sampel 0 pose.

        Returns:
            # results only in this lidar frame
            dic:
                obj_tk_lis (list): Obj token of object in the frame, [N], N>=0
                obj_cate_lis (list): Obj nusc label name, [N]
                obj_cate_id_lis (list): Obj occ label ID, [N]
                static_pc_ref (numpy.ndarray): Static pcd on ref coord, [M, 4]
                static_pc_seg_ref (numpy.ndarray): Static pcd with label on ref
                    coord, [T, 4], [x, y, z, nusc_cate_id]
                obj_pts_lis (list): Obj pcd points, [N, K, 4]
                gt_bbox_3d (numpy.ndarray): Obj bbox, [x, y, z, w, l, h, yaw]
                    x, y, z is minmal coord of this bbox; w, l, h length of
                    x, y, z axis; yaw(rad) bbox direction, y-axis positive
                    direction is 0
                lidar2glb (numpy.ndarray): lidar -> global project matrix, [4, 4]
                lidar_tk (str): lidar token
                is_key_frame (bool): ~
                lidar_file_name (str): e.g. 'xxx.bin.pcd'
        """
        lidar_dic = self.nusc.get('sample_data', lidar_tk)
        lidar_path, boxes, _ = self.nusc.get_sample_data(lidar_dic['token'])

        gt_bbox_3d, obj_tk_lis, obj_cate_lis, obj_cate_id_lis = self.get_bboxes(boxes)

        lidar_path = osp.join(self.cfg.data_root, lidar_dic['filename'])
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[..., :4]

        # got mask of points which is in box [1 N_pt N_box]
        pts_in_boxes = utils.pts_in_bbox(pc, gt_bbox_3d)
        # * cut out movable object points and masks
        # 取出每个bbox中的点
        obj_pts_lis = [pc[pts_in_boxes[0][:, i].bool()]
                       for i in range(pts_in_boxes.shape[-1])]
        # mask for point which is not in any box
        static_pts_mask = ~torch.sum(pts_in_boxes, dim=-1).bool()[0]
        # self car mask
        self_mask = torch.from_numpy(
            (np.abs(pc[:, 0]) > self.cfg.self_range[0]) |
            (np.abs(pc[:, 1]) > self.cfg.self_range[1]) |
            (np.abs(pc[:, 2]) > self.cfg.self_range[2])
        )
        static_pts_mask = static_pts_mask & self_mask
        static_pc = pc[static_pts_mask]

        # convert current lidar point cloud to unified coordinate
        # (frame 0 -- the start frame of the scene)
        lidar2glb_M = self.get_sensor2glb(dic=lidar_dic)['sens2glb']
        lidar2lidar_ref_M = glb2lidar_ref_M @ lidar2glb_M
        static_pc_ref = pc_project(static_pc, lidar2lidar_ref_M)

        if lidar_dic['is_key_frame']:
            seg_lbl = self.load_lidar_seg(lidar_dic['token']).reshape(-1, 1)
            seg_lbl = np.vectorize(self.cfg.labels.nusc_id2occ_id.__getitem__)(seg_lbl)
            pc_seg = np.concatenate([pc[:, :3], seg_lbl], axis=1)
            pc_seg = pc_seg[static_pts_mask]
            static_pc_seg_ref = pc_project(pc_seg, lidar2lidar_ref_M)
        else:
            static_pc_seg_ref = None

        return dict(
            obj_tk_lis=obj_tk_lis,
            obj_cate_lis=obj_cate_lis,
            obj_cate_id_lis=obj_cate_id_lis,
            static_pc_ref=static_pc_ref,
            static_pc_seg_ref=static_pc_seg_ref,
            obj_pts_lis=obj_pts_lis,
            gt_bbox_3d=gt_bbox_3d,
            lidar2glb=lidar2glb_M,
            lidar_tk=lidar_dic['token'],
            is_key_frame=lidar_dic['is_key_frame'],
            lidar_file_name=osp.split(lidar_path)[-1],
        )

    def split_scene(self):
        """划分静态场景和动态目标到两个集合"""
        scene = self.nusc.scene[self.idx]
        tk = scene['token']
        if ((self.cfg.split == 'train' and tk in self.cfg.sampe_val_tk_lis) or
            (self.cfg.split == 'val' and tk not in self.cfg.sampe_val_tk_lis)):
            return

        sensor_tk_lis = self.get_all_lidar_of_scene(scene)

        lidar02glb = self.get_sensor2glb(sensor_tk_lis[0])['sens2glb']

        meta_lis = [self.split_single_lidar_frame(tk, np.linalg.inv(lidar02glb))
                    for tk in sensor_tk_lis]

        static_pc_ref = np.concatenate([dic['static_pc_ref'] for dic in meta_lis])
        static_pc_seg_ref = np.concatenate([dic['static_pc_seg_ref'] for dic in
                                            meta_lis if dic['is_key_frame']])

        # 将所有移动目标合并, 这里的合并是指将每个物体先归一化到自己的相对位置,
        # 即, 左下角为原点, 然后再将多帧中的物体点云进行叠加, 最后得到 id->obj_pts
        obj_tk_lis, obj_cate_id_lis = list(), list()
        for dic in meta_lis:
            Z = [dic['obj_tk_lis'], dic['obj_pts_lis'], dic['obj_cate_id_lis']]
            for obj_tk, obj_pts, cate_id in zip(*Z):
                if obj_tk not in obj_tk_lis and len(obj_pts) > 0:
                    obj_tk_lis.append(obj_tk)
                    obj_cate_id_lis.append(cate_id)

        obj_pts_dic = dict()
        for query_boj_tk in obj_tk_lis:
            query_obj_pts_lis = list()
            for dic in meta_lis:
                Z = [dic['obj_tk_lis'], dic['obj_pts_lis'], dic['gt_bbox_3d']]
                for obj_tk, obj_pts, bbox in zip(*Z):
                    if query_boj_tk == obj_tk and len(obj_pts) > 0:
                        obj_pts = obj_pts[:, :3] - bbox[:3]     # pts - lt_pt
                        rot = Rotation.from_euler('z', -bbox[6], degrees=False)
                        roted_obj_pts = rot.apply(obj_pts)
                        query_obj_pts_lis.append(roted_obj_pts)
            obj_pts_dic[query_boj_tk] = np.concatenate(query_obj_pts_lis, axis=0)

        return dict(
            ref2glb=lidar02glb,
            meta_lis=meta_lis,
            static_pc_ref=static_pc_ref,
            static_pc_seg_ref=static_pc_seg_ref,
            obj_tk_lis=obj_tk_lis,
            obj_cate_id_lis=obj_cate_id_lis,
            obj_pts_dic=obj_pts_dic
        )

    def merge_moving_objects(self, scene_dic, dic):
        # merge moving object to cur lidar frame
        ref2glb = scene_dic['ref2glb']
        obj_tk_lis = scene_dic['obj_tk_lis']
        obj_pts_dic = scene_dic['obj_pts_dic']
        static_pc_ref = scene_dic['static_pc_ref']
        static_pc_seg_ref = scene_dic['static_pc_seg_ref']
        obj_cate_id_lis = np.array(scene_dic['obj_cate_id_lis'], dtype=int)

        # convert static (seg) pc in ref coord -> cur lidar frame
        lidar_uni2lidar = np.linalg.inv(dic['lidar2glb']) @ ref2glb
        _static_pc = pc_project(static_pc_ref, lidar_uni2lidar)[:, :3]  # Nx3
        _static_seg_pc = pc_project(static_pc_seg_ref, lidar_uni2lidar) # Nx4

        lidar_path, boxes, _ = self.nusc.get_sample_data(dic['lidar_tk'])
        gt_bbox_3d, _, _, _ = self.get_bboxes(boxes)

        merge_obj_pts_lis = list()
        merge_obj_seg_lis = list()
        for j, obj_tk in enumerate(dic['obj_tk_lis']):  # j: box_index
            if obj_tk in obj_tk_lis:
                k = obj_tk_lis.index(obj_tk)
                # 合并的动态物体点旋转回当前帧的角度
                obj_pts = obj_pts_dic[obj_tk]
                rot = Rotation.from_euler('z', gt_bbox_3d[:, 6:7][j],degrees=False)
                roted_obj_pts = rot.apply(obj_pts)
                obj_pts = roted_obj_pts + gt_bbox_3d[:, :3][j]
                # 获取在bbox框内的动态物体点云
                if len(obj_pts) > 4:
                    pts_in_boxes = utils.pts_in_bbox(obj_pts, gt_bbox_3d[j: j + 1])
                    obj_pts = obj_pts[pts_in_boxes[0, :, 0].bool()]
                merge_obj_pts_lis.append(obj_pts)
                lbl = obj_cate_id_lis[k][None, ...].repeat(len(obj_pts)).reshape(-1, 1)
                merge_obj_seg_lis.append(np.concatenate([obj_pts[:, :3], lbl], axis=1))

        if merge_obj_pts_lis:
            merge_pc = np.concatenate([_static_pc, *merge_obj_pts_lis])
        else:
            merge_pc = _static_pc
        if merge_obj_seg_lis:
            merge_seg_pc = np.concatenate([_static_seg_pc, *merge_obj_seg_lis])
        else:
            merge_seg_pc = _static_seg_pc

        merge_pc, _ = utils.get_range_mask(merge_pc, range_=self.cfg.pc_range)
        merge_pc = utils.fill_empty_holes(merge_pc, is_calc_normals=True,
                                          max_nn=self.cfg.mesh.max_nn,
                                          depth=self.cfg.mesh.depth,
                                          n_threads=self.cfg.mesh.n_threads,
                                          min_density=self.cfg.mesh.min_density
                                          )
        merge_pc, _ = utils.get_range_mask(merge_pc, range_=self.cfg.pc_range)
        merge_seg_pc, _ = utils.get_range_mask(merge_seg_pc, range_=self.cfg.pc_range)

        return merge_pc, merge_seg_pc

    def save_dense_label(self, pcd, file_name):
        path = osp.join(self.cfg.save_dir, file_name + '.npy')
        os.makedirs(osp.split(path)[0], exist_ok=True)
        np.save(path, pcd)
        logger.info(f"[{self.frame_cnt}] saved at: {path}")

    def restore_occ_dense_label(self):
        scene_dic = self.split_scene()
        meta_lis = scene_dic['meta_lis']

        pc_range = np.array(self.cfg.pc_range)
        voxel_size = np.array(self.cfg.voxel_size)

        for dic in meta_lis:
            if not dic['is_key_frame']:
                continue

            merge_pc, merge_seg_pc = self.merge_moving_objects(scene_dic, dic)
            # voxelize
            voxel_pc = merge_pc.copy()
            voxel_pc = (voxel_pc - pc_range[:3]) / voxel_size
            voxel_pc = np.floor(voxel_pc).astype(int)
            voxel = np.zeros(self.cfg.occ_size)
            voxel[voxel_pc[:, 0], voxel_pc[:, 1], voxel_pc[:, 2]] = 1

            grid = utils.get_grid_index(voxel.shape)
            fov_voxels = grid[voxel > 0]

            fov_voxels[:, :3] = (fov_voxels[:, :3] + voxel_size) * voxel_size
            # 变换到到lidar坐标系
            fov_voxels[:, :3] += pc_range[:3]

            res_pc = utils.get_dense_seg_label(merge_pc, merge_seg_pc)
            # 变换到体素坐标系
            res_pc[:, :3] = (res_pc[:, :3] - pc_range[:3]) / voxel_size
            res_pc = np.floor(res_pc).astype(int)
            logger.debug('Finished converted')
            self.save_dense_label(res_pc, dic['lidar_file_name'])

    def run(self):
        for scene_idx in range(*self.cfg.scene_range):
            self.idx = scene_idx
            self.restore_occ_dense_label()


if __name__ == '__main__':
    from addict import Addict
    from nuscenes import NuScenes
    from tools.gen_occ_nus import config

    config.cfg.update(labels=config.labels)
    cfg = Addict(config.cfg)

    cfg.labels.nusc_name2id = {v: k for k, v in cfg.labels.nusc_id2name.items()}

    _nusc = NuScenes(dataroot=cfg.data_root, version=cfg.version)

    # debug
    # occ_label = OccLabel(_nusc, cfg, idx=0)
    # occ_label.restore_occ_dense_label()

    # run
    occ_label = OccLabel(_nusc, cfg)
    occ_label.run()
