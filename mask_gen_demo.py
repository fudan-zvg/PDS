import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from torch.fft import ifft, fft


save_dir = r'save_dir'

os.makedirs(save_dir,exist_ok=True)
img_num = 200

transform = transforms.Compose([
    transforms.ToTensor(),
])

datasets = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader
data_loader = DataLoader(datasets, batch_size=1, shuffle=True)


def ifft2d(x):
    return ifft(ifft(x, dim=2), dim=3)
def fft2d(x):
    return fft(fft(x, dim=2), dim=3)  ## frequency analysis

j = 0
average_freq_amp = 0
average_amp = 0

for data in data_loader:
    img, label = data
    average_freq_amp += torch.abs(fft2d(img)).mean(dim=0) ** 2
    average_amp += img.mean(dim=0) ** 2
    j += 1
    if j >= img_num:
        break

average_freq_amp /= img_num
average_freq_amp = torch.log(1+average_freq_amp)

average_amp /= img_num

average_freq_amp = average_freq_amp.numpy()
average_amp = average_amp.numpy()

np.save(r'{}/freq_mask'.format(save_dir),average_freq_amp)
np.save(r'{}/space_mask'.format(save_dir),average_amp)

