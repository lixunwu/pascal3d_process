import os

datasetpath = r'D:\dataset\PASCAL3D+_release1.1'
categories = ['aeroplane', 'bicycle', 'boat', 'bottle',
              'bus', 'car', 'chair', 'diningtable',
              'motorbike', 'sofa', 'train', 'tvmonitor']
images = {}
annotations = {}
image_sets = {}
# 设置images,annotations,image_sets的路径
for category in categories:
    annotations[category + '_imagenet'] = os.path.join(datasetpath, 'Annotations', category + '_imagenet')
    annotations[category + '_pascal'] = os.path.join(datasetpath, 'Annotations', category + '_pascal')
    images[category + '_imagenet'] = os.path.join(datasetpath, 'Images', category + '_imagenet')
    images[category + '_pascal'] = os.path.join(datasetpath, 'Images', category + '_pascal')
    image_sets[category + '_imagenet'] = os.path.join(datasetpath, 'Image_sets', category + '_imagenet')
    # image_sets[category + '_pascal'] = os.path.join(datasetpath, 'Image_sets', category + '_pascal')

