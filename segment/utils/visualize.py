import open3d as o3d
import numpy as np


def vis_cate(points, sp_atten, config, save_path):
    pcd = o3d.geometry.PointCloud()
    sp_atten = sp_atten.T
    sp_idx = np.argmax(sp_atten, axis=-1)

    sp_colors = np.random.rand(config.model.superpoint_num, 3)
    colors = sp_colors[sp_idx].reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(save_path, pcd)
