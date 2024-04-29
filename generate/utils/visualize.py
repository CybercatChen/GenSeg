import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


def vis_cate(points, sp_atten, config, save_path):
    pcd = o3d.geometry.PointCloud()
    sp_atten = sp_atten.T
    sp_idx = np.argmax(sp_atten, axis=-1)

    sp_colors = np.random.rand(config.model.superpoint_num, 3)
    colors = sp_colors[sp_idx].reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)


def write_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
