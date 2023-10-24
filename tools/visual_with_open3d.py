import numpy as np
import open3d as o3d
from addict import Addict
from loguru import logger


colors_map = [
    [0, 0, 0, 255],
    [255, 120, 50, 255],    # barrier              orangey
    [255, 192, 203, 255],   # bicycle              pink
    [255, 255, 0, 255],     # bus                  yellow
    [0, 150, 245, 255],     # car                  blue
    [0, 255, 255, 255],     # construction_vehicle cyan
    [200, 180, 0, 255],     # motorcycle           dark orange
    [255, 0, 0, 255],       # pedestrian           red
    [255, 240, 150, 255],   # traffic_cone         light yellow
    [135, 60, 0, 255],      # trailer              brown
    [160, 32, 240, 255],    # truck                purple
    [255, 0, 255, 255],     # driveable_surface    dark pink
    # [175,   0,  75, 255], # other_flat           dark red
    [139, 137, 137, 255],
    [75, 0, 75, 255],       # sidewalk             dard purple
    [150, 240, 80, 255],    # terrain              light green
    [230, 230, 250, 255],   # manmade              white
    [0, 175, 0, 255],       # vegetation           green
    [0, 255, 127, 255],     # ego car              dark cyan
    [255, 99, 71, 255],
    [0, 191, 255, 255],
    # [175,   0,  75, 255], # other_flat           dark red
]


class VoxelBox:
    """ Generate open3d cube box
        7--------6
       /|       /|
      / |      / |
     /  |3    /  2
    4--------5  /
    |  /     | /
    | /      |/
    0--------1

    coordinate:
    z         y
    |       .
    |    .
    | .
    0---------x
    """
    # 8 cube vertice, order as above
    _vertice = np.stack([[-1, 1, 1, -1, -1, 1, 1, -1],
                         [-1, -1, 1, 1, -1, -1, 1, 1],
                         [-1, -1, -1, -1, 1, 1, 1, 1]], axis=1)
    # connection between vertices
    _edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4],
                      [0, 4], [1, 5], [2, 6], [3, 7]])

    @classmethod
    def get_pc_voxel_box(cls, pc, voxel_size):
        """ Get point cloud voxel vertices and their connection

        Args:
            pc (np.array): [N, 4] or [N, 3], [x,y,z, (cls)]
            voxel_size (np.array): [x, y, z] size
        """
        pc_num = len(pc)
        offset = cls._vertice * (voxel_size / 2.)
        offset = np.tile(offset[None, ...], (pc_num, 1, 1))
        vertice = offset + pc[:, :3][:, None, :]

        vertice_group = np.arange(0, pc_num * 8, 8)
        edge = np.tile(cls._edge[None, ...], (pc_num, 1, 1))
        edge += vertice_group[:, None, None]

        return vertice, edge

    @classmethod
    def test(cls):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window('Test')
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(cls._vertice.reshape((-1, 3)))
        line_sets.lines = o3d.open3d.utility.Vector2iVector(cls._edge.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

        vis.run()
        vis.destroy_window()
        del vis


class RenderPointCloud:
    def __init__(self, fpath, cfg, mode='pcd'):
        """
        Args:
            mode (str):
                'pcd': point cloud, [x, y, z, (cls)]
                'voxel': voxel, [H, W, Z] = (cls)
                'mesh': ~
        """
        self.pc = np.load(fpath)
        self.cfg = Addict(cfg)
        self.vis = self.init_vis()
        if self.cfg.is_voxel:
            self.voxel_size = np.array(self.cfg.voxel_size)
        logger.debug(f"Loaded pcd shape: {self.pc.shape}")

    def init_vis(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(self.cfg.window_title)
        opt = vis.get_render_option()
        opt.background_color = self.cfg.background_color
        return vis

    def get_pc_voxel_box(self, pc):
        vertice, edge = VoxelBox.get_pc_voxel_box(pc, self.voxel_size)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.open3d.utility.Vector3dVector(
            vertice.reshape((-1, 3)))
        line_set.lines = o3d.open3d.utility.Vector2iVector(
            edge.reshape((-1, 2)))
        line_set.paint_uniform_color((0, 0, 0))
        return line_set

    def simple_render(self):
        label = self.pc[:, 3]
        assert label.max() < len(self.cfg.colors_map)

        colors = self.cfg.colors_map[label.astype(int)]
        print('类别数量', len(np.unique(label)))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pc[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pcd])

    def render(self, is_simple=False):
        label = self.pc[:, 3]
        assert label.max() < len(self.cfg.colors_map), f"{label.max()}"

        colors = self.cfg.colors_map[label.astype(int)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pc[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.6, origin=[0, 0, 0])

        self.vis.add_geometry(mesh_frame)

        # notice: only support CUBE! so, only can give A size for voxel_size
        vox_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,voxel_size=self.voxel_size[0])
        self.vis.add_geometry(vox_grid)

        line_set = self.get_pc_voxel_box(self.pc)
        self.vis.add_geometry(line_set)

        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    path = '__tmp/dense_voxel.npy'

    cfg = dict(
        window_title='demo',
        colors_map=np.array(colors_map) / 255,
        background_color=[1, 1, 1],
        is_disp_as_voxel=True,
        is_voxel=True,
        voxel_size=[0.5, 0.5, 0.5],
    )

    R = RenderPointCloud(path, cfg)
    R.render()
