import torch
from easydict import EasyDict
import torch.nn.functional as F


def sample_aprt_point(args, labels, points, desired_points_per_class):
    part_points = []

    for class_id in range(args.part_num):
        class_mask = (labels == class_id)
        point_copy = points.clone()
        class_points = torch.zeros((args.batch_size, desired_points_per_class, 3))
        point_copy[~class_mask] = 0
        points_count = torch.sum(class_mask, dim=1)

        undersampled_mask = points_count < desired_points_per_class
        oversampled_mask = points_count > desired_points_per_class

        for j in range(args.batch_size):
            if points_count[j] == 0:
                class_points[j] = torch.zeros((desired_points_per_class, 3))
            else:
                if undersampled_mask[j]:
                    non_zero_indices = torch.nonzero(class_mask[j]).squeeze(1)
                    random_indices = torch.randint(0, non_zero_indices.size(0), (desired_points_per_class,),
                                                   device=non_zero_indices.device)
                    selected_indices = non_zero_indices[random_indices]
                    class_points[j] = points[j, selected_indices]

                elif oversampled_mask[j]:
                    non_zero_indices = torch.nonzero(class_mask[j]).squeeze(1)
                    random_indices = torch.randint(0, non_zero_indices.size(0), (desired_points_per_class,),
                                                   device=non_zero_indices.device)
                    selected_indices = non_zero_indices[random_indices]
                    class_points[j] = points[j, non_zero_indices[random_indices]]

        part_points.append(class_points)

    return part_points


def sample_point(args, labels, points, desired_points_per_class=400):
    B, E, N = points.shape
    one_hot_labels = F.one_hot(labels, num_classes=args.part_num).float()  # [B, N, part_num]
    one_hot_labels = one_hot_labels.transpose(1, 2)  # [B, part_num, N]
    part_point_feat = torch.einsum('bcd,bed->bcde', one_hot_labels, points)  # [B, part_num, N, 3]

    points_count = torch.sum(one_hot_labels, dim=2)  # [B, part_num]
    sampled_points = torch.zeros((B, args.part_num, desired_points_per_class, E), device=points.device)

    sampling_needed = points_count != 0

    for b in range(B):
        for p in range(args.part_num):
            if sampling_needed[b, p]:
                non_zero_indices = torch.nonzero(one_hot_labels[b, p]).squeeze(1)
                num_points = non_zero_indices.size(0)
                if num_points <= desired_points_per_class:
                    indices = torch.randint(0, num_points, (desired_points_per_class,), device=points.device)
                else:
                    indices = torch.randperm(num_points, device=points.device)[:desired_points_per_class]

                sampled_points[b, p] = part_point_feat[b, p, non_zero_indices[indices]]

    return sampled_points


class AverageMeter(object):
    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values):
        if type(values).__name__ == 'list':
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v
                self._count[idx] += 1
        else:
            self._val[0] = values
            self._sum[0] += values
            self._count[0] += 1

    def val(self, idx=None):
        if idx is None:
            return self._val[0] if self.items is None else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return self._count[0] if self.items is None else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return self._sum[0] / self._count[0] if self.items is None else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]


def print_log(args, string):
    with open(args.log_file, 'a') as f:
        f.write(string + "\n")
    print(string)


def log_config_to_file(args, config, pre='config'):
    for key, val in config.items():
        if isinstance(config[key], EasyDict):
            print_log(args, f'{pre}.{key} = easydict()')
            log_config_to_file(args, config[key], pre=pre + '.' + key)
            continue
        print_log(args, f'{pre}.{key} : {val}')
