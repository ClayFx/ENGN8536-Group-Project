import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2


transform_no_training = T.Compose([
        T.ToPILImage(),
        # T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

transform_training = T.Compose([
        T.ToPILImage(),
        T.Resize((256,256)),
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomCrop(512, padding=4),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

class VideoDataset(Dataset):
    def __init__(self, root, label_root, transforms=transform_training, train=True):
        self.train = train
        videos = os.listdir(root)
        self.videos = [os.path.join(root, video) for video in videos]
        self.transforms = transforms
        self.label_root = label_root

    def __getitem__(self, index):
        video_path = self.videos[index]
        imgs = [cv2.imread(os.path.join(video_path, img)) for img in sorted(os.listdir(video_path))[-3:-1]]
        if self.transforms:
            imgs = [self.transforms(img) for img in imgs]
        imgs = np.stack(imgs)
        # Generating labels
        label_filename = os.path.join(self.label_root, video_path.split('\\')[-1])
        label = cv2.imread(os.path.join(label_filename, sorted(os.listdir(label_filename))[-1]))
        label = handle_label_ce(label)
        label = cv2.resize(label,(256,256))
        label= np.transpose(label, (2, 0, 1))
        return imgs, label_filename, label

    def __len__(self):
        return len(self.videos)
    
def handle_label_mse(label):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors = [tuple(color) for color in colors]

    h, w, c = label.shape
    assert c == 3
    new_label = np.zeros((h, w, 19))
    for i in range(h):
        for j in range(w):
            if tuple(label[i,j]) in colors:
                new_label[i, j, colors.index(label[i,j])] = 1
            else:
                new_label[i, j, 18] = 1
    return new_label

def handle_label_ce(label):
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors = [tuple(color) for color in colors]

    h, w, c = label.shape
    assert c == 3
    new_label = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if tuple(label[i,j]) in colors:
                new_label[i, j] = colors.index(label[i,j])
            else:
                new_label[i, j] = 18
    return new_label


if __name__ == '__main__':
    """
    This is to test whether the custom dataset is well-defined
    """
    from torch.utils.data import DataLoader
    # Local Data path
    PATH = "../data/image"
    LABEL_PATH = "../data/label"

    # Read the dataset
    torch_data = VideoDataset(PATH, LABEL_PATH, transforms=transform_training)
    datas = DataLoader(torch_data, batch_size=1, shuffle=False, drop_last=False, num_workers=2)


    for i, data in enumerate(datas):
        # print("The {}-th Batch \n{}".format(i, data))
        # # data
        # print(data[0])
        # # label
        # print(data[1])
        imgs, label_filename, labels = data
        print(imgs.size(), label_filename,labels.size())