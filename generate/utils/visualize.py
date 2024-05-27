import numpy as np
from plyfile import PlyData, PlyElement


def vis_cate(points, label, args, save_path):
    np.random.seed(seed=2024)
    sp_colors = np.random.rand(args.part_num, 3)
    colors = sp_colors[label].reshape(-1, 3)

    if np.max(colors) > 1 or np.min(colors) < 0:
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

    colors = (colors * 255).astype(int)

    points = [(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]) for i in
              range(points.shape[0])]
    vertex = np.array(points,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el_vertex = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el_vertex], text=True).write(save_path)


def write_ply(filename, points):
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)
