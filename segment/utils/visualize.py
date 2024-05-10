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


def write_ply_with_color(filename, points_list):
    np.random.seed(seed=2024)
    sp_colors = np.random.rand(len(points_list), 3)

    all_points = np.concatenate(points_list, axis=0)
    all_colors = np.concatenate(
        [np.tile(sp_colors[idx], (points.shape[0], 1)) for idx, points in enumerate(points_list)], axis=0)

    if np.max(all_colors) > 1 or np.min(all_colors) < 0:
        all_colors = (all_colors - np.min(all_colors)) / (np.max(all_colors) - np.min(all_colors))
    all_colors = (all_colors * 255).astype(int)

    colored_points = [
        (all_points[i, 0], all_points[i, 1], all_points[i, 2], all_colors[i, 0], all_colors[i, 1], all_colors[i, 2]) for
        i in range(all_points.shape[0])]
    vertex = np.array(colored_points,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el_vertex = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el_vertex], text=True).write(filename)