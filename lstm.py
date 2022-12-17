import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from read_data import build_data_set, get_vid_path


class FrameSeq(Dataset):
    def __init__(self, split, root=os.sep.join([
            os.path.dirname(__file__),
            'VIDEOS'
            ]), window_size=4, transform=None):
        self.transform = transform
        path = get_vid_path()
        X = build_data_set(path)
        if (split == 'train'):
            self.data = X[:40]
        elif (split == 'test'):
            self.data = X[40:]
        else:
            raise ValueError('split must be "train" or "test"')
        self.before = []
        self.after = []
        num_sample, num_frame, _, _, _ = self.data.shape
        for i in range(num_sample):
            for j in range(num_frame - 2*window_size + 1):
                self.before.append(self.data[i, j:j+4])
                self.after.append(self.data[i, j+4:j+8])
        # self.data = torch.from_numpy(self.data).permute(0, 1, 4, 2, 3)
        self.before = torch.from_numpy(
            np.stack(self.before, axis=0)
        ).permute(0, 1, 4, 2, 3)
        self.after = torch.from_numpy(
            np.stack(self.after, axis=0)
        ).permute(0, 1, 4, 2, 3)

    def __getitem__(self, idx):
        before_item = self.before[idx].type(torch.float)
        after_item = self.after[idx].type(torch.float)
        if self.transform is not None:
            before_item = self.transform(before_item)
            after_item = self.transform(after_item)
        # print(before_item.shape)
        return (before_item, after_item)

    def __len__(self):
        return len(self.before)


class LSTMModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=16, layer_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_encode = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm_decode = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True)
        self.act = nn.LeakyReLU(0.2)
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(
            self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        x0 = torch.zeros_like(x)
        x = torch.cat([x, x0], dim=1)
        # print(x.shape)
        out, (hn, cn) = self.lstm_encode(x, (h0.detach(), c0.detach()))
        # print(hn.mean(), cn.mean())
        out = out[:, 4:]
        # print('out', out.max())
        out = self.act(self.linear1(out))
        out = self.act(self.linear2(out))
        # print('res', out.max())
        # print(out.shape)
        return out
        # print(out_pred.shape)
        # return out_pred
        # return self.out(self.linear(self.act(out_pred)))


class FF(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        return x
