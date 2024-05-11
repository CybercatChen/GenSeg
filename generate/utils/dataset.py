from copy import copy
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from plyfile import PlyData, PlyElement
import os

synsetid_to_cate = {
    '517517': 'vessel',
    '02691156': 'airplane',
    '03001627': 'chair',
    '594666': 'vessel_left',
    '985211': 'vessel_all'
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class PCDataset(Dataset):

    def __init__(self, data_path, output_path, cates, split, scale_mode, raw_data=None, transform=None):
        super().__init__()
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.data_path = data_path
        self.output_path = os.path.join(output_path, cates + '.hdf5')
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

    def convert_to_hdf5(self, train_ratio=0.8):

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
            if pcd.shape[0] < self.sample_num:
                continue
            if self.sample_num > 0:
                choice = np.random.choice(pcd.shape[0], self.sample_num, replace=False)
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

        basename = os.path.basename(self.output_path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.output_path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)
        stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

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
        torch.save(self.stats, stats_save_path)
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
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, pc_dim)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, pc_dim)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, pc_dim)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, pc_dim)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
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
