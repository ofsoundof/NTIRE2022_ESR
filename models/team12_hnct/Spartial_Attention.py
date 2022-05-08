
import torch
from torch import nn
import torch.nn.functional as F


class Spartial_Attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(Spartial_Attention, self).__init__()

        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2

        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)

        mask = self.__layer(mask)
        y = x * mask
        return y

if __name__ == '__main__':
    model = Spartial_Attention()
    x = torch.randn((16, 50, 48, 48))
    x1 = model(x)
    print(x1.shape)