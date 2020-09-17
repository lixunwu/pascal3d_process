import os
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import pymesh

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

imagenet_pca = {
    'eigval': torch.tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.tensor([[-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203]])}


# 根据图像的bbox随机裁剪
def random_crop(img, x, y, w, h):
    left = max(0, x + int(w * np.random.uniform(low=-0.1, high=0.1)))
    upper = max(0, y + int(h * np.random.uniform(low=-0.1, high=0.1)))
    right = min(img.size[0], x + int(w * np.random.uniform(low=0.9, high=1.1)))
    lower = min(img.size[1], x + int(h * np.random.uniform(low=0.9, high=1.1)))
    img_crop = img.crop((left, upper, right, lower))
    return img_crop


# resize到dim*dim
def resize_pad(img, dim):
    w, h = img.size
    # torchvison 只接收PIL图像,将最小边长缩放到给定的size,最大边长也按照等比例缩放
    img = transforms.functional.resize(img, int(dim * (min(w, h) / max(w, h))))
    left_pad = int(np.ceil((dim - img.size[0]) / 2))
    right_pad = int(np.floor((dim - img.size[0]) / 2))
    top_pad = int(np.ceil((dim - img.size[1]) / 2))
    bottom_pad = int(np.floor((dim - img.size[1]) / 2))
    img = transforms.functional.pad(img, (left_pad, top_pad, right_pad, bottom_pad))
    return img


# Lighting noise transform
class TransLightning(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


def read_pointcloud(model_path, point_num):
    raw_data = pymesh.load_mesh(model_path).vertices
    point_subset = np.random.choice(raw_data.shape[0], point_num, replace=False)
    point_cloud = raw_data[point_subset]
    point_cloud = torch.from_numpy(point_cloud.transpose()).float()

    # 归一化到[0,1]
    point_cloud = point_cloud - torch.min(point_cloud)
    point_cloud = point_cloud / torch.max(point_cloud)
    return point_cloud


class Pascal3D(data.Dataset):
    def __init__(self, root_dir, annotation_file, input_dim=224, point_num=2500,
                 train=True, keypoint=True, novel=True, cls_choice=None, shot=None):
        self.train = train
        self.root_dir = root_dir
        self.input_dim = input_dim
        self.point_num = point_num

        # 从annotation_file(Pascal3D.txt)中读取数据
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        # 将ele=90和difficult的数据清除
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]

        if train:
            frame = frame[frame.set == 'train']

        else:
            # 只使用没有遮挡和截断且有关键点的数据来进行验证
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]
            frame = frame[frame.has_keypoints == 1]

        # 在训练数据中 如果keypoint==True 则只选择有keypoint的数据进行训练
        if train and keypoint:
            frame = frame[frame.has_keypoints == 1]

        self.annotation_frame = frame
        self.im_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
        ])
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.render_transform = transforms.ToTensor()
        pass

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotation_frame.iloc[idx]['im_path'])
        cls = self.annotation_frame.iloc[idx]['cat']

        # 随机选择一个shape
        cls_index = np.unique(self.annotation_frame[self.annotation_frame.cat == cls].cad_index)
        cad_index = np.random.choice(cls_index)

        # bbox
        left = self.annotation_frame.iloc[idx]['left']
        upper = self.annotation_frame.iloc[idx]['upper']
        right = self.annotation_frame.iloc[idx]['right']
        lower = self.annotation_frame.iloc[idx]['lower']

        # viewpoint
        viewpoint = self.annotation_frame.iloc[idx, 9:12].values

        # 读取图像CHW RGB
        img = Image.open(img_path).convert('RGB')

        # 如果数据集是用来训练的,则做一些增强操作
        if self.train:
            # 高斯模糊
            if min(right - left, lower - upper) > 224 and np.random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(3))

            # 使用bbox随机裁剪
            img = random_crop(img=img, x=left, y=upper, w=right - left, h=lower - upper)

            # 水平翻转
            if np.random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                viewpoint[0] = 360 - viewpoint[0]
                viewpoint[2] = -viewpoint[2]

            # 随机旋转
            if np.random.random() > 0.5:
                r = max(-60, min(60, np.random.randn() * 30))
                img.rotate(r)
                viewpoint[2] = viewpoint[2] + r
                viewpoint[2] += 360 if viewpoint[2] < -180 else (-360 if viewpoint[2] > 180 else 0)

            # 将图片resize到固定大小(224*224)
            img = resize_pad(img, self.input_dim)
            img = self.im_augmentation(img)

        # 如果不是用来训练的,则不进行数据增强操作,只进行crop和resize
        else:
            img = img.crop((left, upper, right, lower))
            img = resize_pad(img, self.input_dim)
            img = self.im_transform(img)

        # 将viewpoint 转换为[0,360) 没有必要?
        # viewpoint[0] = (360. - viewpoint[0]) % 360.
        # viewpoint[1] = viewpoint[1] + 90.
        # viewpoint[2] = (viewpoint[2] + 180.) % 360.
        viewpoint = viewpoint.astype('int')

        # load point_clouds
        point_cloud_path = os.path.join(
            self.root_dir, 'Pointclouds', cls, '{:02d}'.format(cad_index), 'compressed.ply')
        point_cloud = read_pointcloud(point_cloud_path, self.point_num)

        return {'img': img, 'point_cloud': point_cloud, 'label': viewpoint, 'class': cls}


if __name__ == '__main__':
    pascal3d = Pascal3D(root_dir='/opt/WLX/dataset/PASCAL3D+_release1.1',
                        annotation_file='Pascal3D.txt')
    data = pascal3d[0]
    print('end')
