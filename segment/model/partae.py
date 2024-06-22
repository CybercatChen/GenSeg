import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config is None:
            self.input_dim = 3
            self.output_dim = [512, 256]
            self.use_global = True
            self.hidden_dim = [64, 256, 1024]
        else:
            self.input_dim = config.model.encoder.input_dim
            self.output_dim = config.model.encoder.output_dim
            self.use_global = config.model.encoder.use_global
            self.hidden_dim = config.model.encoder.hidden_dim

        assert len(self.hidden_dim) > 0
        assert len(self.output_dim) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim[0], 1),
            nn.BatchNorm1d(self.hidden_dim[0]),
            # nn.LeakyReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.hidden_dim[0]
        for c in self.hidden_dim[1:]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                # nn.LeakyReLU(inplace=True)
            ))
            hidden_in = c

        self.final_layer = nn.Sequential()
        if self.use_global:
            hidden_in = self.hidden_dim[-1] + self.hidden_dim[0]
        else:
            hidden_in = self.hidden_dim[-1]
        for c in self.output_dim:
            self.final_layer.add_module(str(len(self.final_layer)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                # nn.LeakyReLU(inplace=True)
            ))
            hidden_in = c

    def forward(self, points):
        """
        forward of encoder
        :param points: input data, B(batch) N(num) 3(xyz input)
        :return: point-wise feature, B N C
        """
        B, N, C = points.shape
        points = points.transpose(2, 1)  # B 3 N

        x = self.input_layer(points)  # B 64(or other) N
        px = x  # B 64(or other) N

        for layer in self.hidden_layers:
            x = layer(x)  # B _ N

        if self.use_global:
            x_global = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, N)  # B 1024(or other) N
            x = torch.cat([x_global, px], dim=1)
        x = self.final_layer(x).transpose(2, 1)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config is None:
            self.input_dim = [256, 512]
            self.output_dim = 3
            self.use_global = True
            self.hidden_dim = [1024, 256, 64]
        else:
            self.input_dim = config.model.encoder.output_dim
            self.output_dim = 3
            self.use_global = config.model.encoder.use_global
            self.hidden_dim = config.model.encoder.hidden_dim
            self.point_num = 2048
        assert len(self.hidden_dim) > 0
        assert len(self.input_dim) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_dim[-1], self.input_dim[0], 1),
            nn.BatchNorm1d(self.input_dim[0]),
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.input_dim[0]
        for c in self.hidden_dim[::-1]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
            ))
            hidden_in = c

        self.final_layer = nn.Sequential(
            nn.Conv1d(self.hidden_dim[0], self.output_dim, 1),
            nn.BatchNorm1d(self.output_dim),
        )

        self.point_layer1 = nn.Sequential(
            nn.Conv1d(4, self.point_num // 2, 1),
        )

        self.point_layer2 = nn.Sequential(
            nn.Conv1d(self.point_num // 2, self.point_num, 1),
        )

    def forward(self, features):
        x = self.input_layer(features)  # B 512 N
        for layer in self.hidden_layers:
            x = layer(x)  # B _ N
        x = self.final_layer(x).transpose(2, 1)  # B N 3
        x = self.point_layer1(x)
        x = self.point_layer2(x)
        return x
