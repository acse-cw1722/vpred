import os
import torch
from torch.utils.data import Dataset
from read_data import get_vid_path, build_data_set
from torch import nn


class VideoFrame(Dataset):
    def __init__(self, split, root=os.sep.join([
            os.path.dirname(__file__),
            'VIDEOS'
            ]), transform=None):
        self.transform = transform
        path = get_vid_path()
        X = build_data_set(path)
        if (split == 'train'):
            self.data = X[:40]
        elif (split == 'test'):
            self.data = X[40:]
        else:
            raise ValueError('split must be "train" or "test"')
        self.data = torch.from_numpy(self.data)
        self.data = torch.stack([
            image for video in self.data for image in video
        ], dim=0)

    def __getitem__(self, idx):
        item = self.data[idx]
        item = item.permute(2, 1, 0)
        item = item.type(torch.float)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.data)


class Encoder(nn.Module):
    def __init__(
        self, input_size=(3, 128, 128), hidden_dim_1=32, hidden_dim_2=64,
        hidden_dim_3=64, code_dim=30
    ):
        super().__init__()
        self.input_channels = input_size[0]
        self.resolution = input_size[1]
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels, out_channels=hidden_dim_1,
                kernel_size=(5, 5), padding=2, stride=(2, 2)
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim_1, out_channels=hidden_dim_2,
                kernel_size=(5, 5), padding=2, stride=(2, 2)
            ),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim_2, out_channels=hidden_dim_3,
                kernel_size=(3, 3), padding=1, stride=(2, 2)
            ),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.feedforward = nn.Sequential(
            nn.Linear(
                in_features=int(self.resolution / 8)**2 * hidden_dim_3,
                out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=code_dim)
        )

    def forward(self, x: torch.Tensor):
        # print('input', x.shape)
        x = self.conv1(x)
        # print('conv1(x)', x.shape)
        x = self.conv2(x)
        # print('conv2(x)', x.shape)
        x = self.conv3(x)
        # print('conv3(x)', x.shape)
        x = self.flatten(x)
        x = self.feedforward(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, input_size=(3, 128, 128), hidden_dim_1=32, hidden_dim_2=64,
        hidden_dim_3=64, code_dim=30
    ):
        super().__init__()
        self.input_channels = input_size[0]
        self.resolution = input_size[1]
        self.feedforward = nn.Sequential(
            nn.Linear(
                in_features=code_dim,
                out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=int(self.resolution / 8)**2 * hidden_dim_3
            )
        )
        self.unflattend = nn.Unflatten(
            dim=1, unflattened_size=(
                hidden_dim_3,
                int(self.resolution / 8), int(self.resolution / 8)
            )
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim_3, out_channels=hidden_dim_2,
                kernel_size=(2, 2), stride=(2, 2), output_padding=0
            ),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim_2, out_channels=hidden_dim_1,
                kernel_size=(2, 2), stride=(2, 2), output_padding=0
            ),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_dim_1, out_channels=self.input_channels,
                kernel_size=(2, 2), stride=(2, 2), output_padding=0
            ),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        # print('input shape', x.shape)
        x = self.feedforward(x)
        # print('ffd(x)', x.shape)
        x = self.unflattend(x)
        # print('unflatten(x)', x.shape)
        x = self.deconv3(x)
        # print('deconv3(x)', x.shape)
        x = self.deconv2(x)
        # print('deconv2(x)', x.shape)
        x = self.deconv1(x)
        # print('deconv1(x)', x.shape)
        return x
