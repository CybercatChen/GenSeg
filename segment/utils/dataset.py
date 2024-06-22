from copy import copy
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from plyfile import PlyData, PlyElement
import os

synsetid_to_cate = {
    '517517': 'vessel',
    '985211': 'vessel_all',
    '594666': 'vessel_left',
    '02691156': 'airplane',
    '03001627': 'chair',
    '03648465': 'skeleton_left',
    '03648146': 'skeleton_left_1000'
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class PCDataset(Dataset):

    def __init__(self, data_path, cates, split, scale_mode, raw_data=None, transform=None):
        super().__init__()
        self.data_path = os.path.join(data_path, cates + '.hdf5')
        self.cate_synsetids = cate_to_synsetid[cates]
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.sample_num = 1000
        self.pointclouds = []
        self.stats = None

        if raw_data is not None:
            self.raw_data = raw_data
            self.convert_to_hdf5()
        self.get_statistics()
        self.load()

    def convert_to_hdf5(self, train_ratio=1):

        def read_ply(inputfile):
            plydata = PlyData.read(inputfile)
            pcd = np.zeros((plydata['vertex']['x'].shape[0], 3))
            pcd[:, 0] = plydata['vertex']['x']
            pcd[:, 1] = plydata['vertex']['y']
            pcd[:, 2] = plydata['vertex']['z']
            return pcd

        inputfiles = [os.path.join(self.raw_data, f) for f in os.listdir(self.raw_data) if f.endswith('.ply')]
        h5file = h5py.File(self.data_path, 'w')
        train_num = int(len(inputfiles) * train_ratio)
        train_pcds = []
        val_pcds = []
        pid = 0
        print("initial dataset...")
        for inputfile in inputfiles:
            pcd = read_ply(inputfile)
            point_num = pcd.shape[0]
            if point_num != 0:
                if point_num < self.sample_num:
                    choice = np.random.choice(point_num, self.sample_num, replace=True)
                    pcd = pcd[choice]
                elif point_num > self.sample_num:
                    choice = np.random.choice(point_num, self.sample_num, replace=False)
                    pcd = pcd[choice]
                if pid < train_num:
                    train_pcds.append(pcd)
                else:
                    val_pcds.append(pcd)
                pid += 1

        print('train:', np.array(train_pcds).shape)
        print('val:', np.array(val_pcds).shape)
        cate_g = h5file.create_group(self.cate_synsetids)
        cate_g.create_dataset('train', data=np.array(train_pcds), dtype='f')
        cate_g.create_dataset('val', data=np.array(val_pcds), dtype='f')

    def get_statistics(self):
        with h5py.File(self.data_path, 'r') as f:
            pointclouds = []

            for split in ('train', 'val'):
                print("---------------------------------------")
                print(torch.from_numpy(f[self.cate_synsetids][split][...]).type)
                pointclouds.append(torch.from_numpy(f[self.cate_synsetids][split][...]).type(torch.FloatTensor))

        all_points = torch.cat(pointclouds, dim=0)  # (B, N, 3)
        B, N, _ = all_points.size()
        print('size:', B)
        mean = all_points.view(B * N, -1).mean(dim=0)  # (1, 3)
        std = all_points.view(-1).std(dim=0)  # (1, )

        self.stats = {'mean': mean, 'std': std}
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            cate_name = synsetid_to_cate[self.cate_synsetids]
            for j, pc in enumerate(f[self.cate_synsetids][self.split]):
                yield torch.from_numpy(pc), j, cate_name

        with h5py.File(self.data_path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):
                pc_dim = pc.shape[1]
                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, pc_dim)
                    scale = self.stats['std'].reshape(1, 1)
                else:
                    shift = torch.zeros([1, pc_dim])
                    scale = torch.ones([1, pc_dim])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data


class PartDataset(Dataset):
    def __init__(self, data_path, cates, split, scale_mode='global_unit', raw_data=None, transform=None):
        super().__init__()
        self.data_path = os.path.join(data_path, cates + '.hdf5')
        self.cate_synsetids = cate_to_synsetid[cates]
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.sample_num = 2048
        self.pointclouds = []
        self.stats = None

        if raw_data is not None:
            self.raw_data = raw_data
            self.convert_to_hdf5()
        self.get_statistics()
        self.load()

    def convert_to_hdf5(self):
        def read_pts_file(file_path):
            points = np.loadtxt(file_path)
            return points

        def read_seg_file(file_path):
            with open(file_path, 'r') as file:
                segments = [int(line.strip()) for line in file]
            return segments

        pts_files = [f for f in os.listdir(os.path.join(self.raw_data, 'data')) if f.endswith('.pts')]
        seg_files = [f for f in os.listdir(os.path.join(self.raw_data, 'label')) if f.endswith('.seg')]

        assert len(pts_files) == len(seg_files), "Mismatch between number of .pts and .seg files"

        h5file = h5py.File(self.data_path, 'w')
        pcds, labels = [], []

        for pts_file, seg_file in zip(pts_files, seg_files):
            pts_path = os.path.join(self.raw_data, 'data', pts_file)
            seg_path = os.path.join(self.raw_data, 'label', seg_file)

            points = read_pts_file(pts_path)
            segments = read_seg_file(seg_path)

            if points.shape[0] < self.sample_num:
                continue
            if self.sample_num > 0:
                choice = np.random.choice(points.shape[0], self.sample_num, replace=False)
                points = points[choice]
                segments = np.array(segments)[choice]

            pcds.append(points)
            labels.append(segments)

        cate_g = h5file.create_group(self.cate_synsetids)
        cate_g.create_dataset('train_points', data=np.array(pcds), dtype='f')
        cate_g.create_dataset('train_labels', data=np.array(labels), dtype='i')
        h5file.close()

    def get_statistics(self):
        with h5py.File(self.data_path, 'r') as f:
            pointclouds = torch.from_numpy(f[self.cate_synsetids]['train_points'][...]).type(torch.FloatTensor)

        B, N, _ = pointclouds.size()
        mean = pointclouds.view(B * N, -1).mean(dim=0)  # (1, 3)
        std = pointclouds.view(-1).std(dim=0)  # (1, )
        self.stats = {'mean': mean, 'std': std}
        return self.stats

    def load(self):
        def _enumerate_pointclouds(f):
            for j, pc in enumerate(f[self.cate_synsetids]['train_points']):
                labels = f[self.cate_synsetids]['train_labels'][j]
                yield torch.from_numpy(pc), torch.from_numpy(labels), j

        with h5py.File(self.data_path, mode='r') as f:
            for pc, labels, pc_id in _enumerate_pointclouds(f):
                pc_dim = pc.shape[1]
                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, pc_dim)
                    scale = self.stats['std'].reshape(1, 1)
                else:
                    shift = torch.zeros([1, pc_dim])
                    scale = torch.ones([1, pc_dim])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'labels': labels,
                    'cate': self.cate_synsetids,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k: v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data
