import os
import scipy.io as sio
from tqdm import trange

dataset_path = '/opt/WLX/dataset/PASCAL3D+_release1.1'
annotation_file = 'Pascal3D.txt'
cats = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
        'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
databases = ['imagenet', 'pascal']
subsets = ['train', 'val', 'test']

if not os.path.exists(annotation_file):
    with open(os.path.join(dataset_path, annotation_file), 'w') as f:
        f.write(
            'source,set,cat,im_name,object,cad_index,truncated,occluded,difficult,azimuth,elevation,'
            'inplane_rotation,left,upper,right,lower,has_keypoints,distance,im_path\n')

for database in databases:
    for cat in cats:
        for subset in subsets:
            # 无论是'imagenet'还是'pascal',图片数据和标注数据都在'./Images'和'./Annotations'下
            image_dir = os.path.join(dataset_path, 'Images', cat + '_' + database)
            annotation_dir = os.path.join(dataset_path, 'Annotations', cat + '_' + database)

            # 收集当前subset下的数据列表
            if database == 'imagenet':
                if subset == 'train':
                    subset_list_file = os.path.join(
                        dataset_path, 'Image_sets', f'{cat}_imagenet_train.txt')
                if subset == 'val':
                    subset_list_file = os.path.join(
                        dataset_path, 'Image_sets', f'{cat}_imagenet_val.txt')
                if subset == 'test':
                    continue
            if database == 'pascal':
                if subset == 'train':
                    subset_list_file = os.path.join(
                        dataset_path,
                        'PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main',
                        f'{cat}_train.txt')
                if subset == 'val':
                    subset_list_file = os.path.join(
                        dataset_path,
                        'PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main',
                        f'{cat}_val.txt')
                if subset == 'test':
                    subset_list_file = os.path.join(
                        dataset_path,
                        'PASCAL', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main',
                        f'{cat}_test.txt')

            with open(subset_list_file, 'r') as f:
                # f.read() 返回一个str,包含'\n',str.splitlines()将str按照'\n'分割成list
                subset_list = f.read().splitlines()

            # 在Pascal的val.txt数据中,每行的数据格式为'2008_000002 -1',
            # 而imagenet的val.txt数据中,每行的数据格式为'n02690373_1002',
            # 所以要去除Pascal val_list中的classification label(-1)
            if database == 'pascal':
                subset_list = [str(line.split()[0]) for line in subset_list]
            subset_list.sort()
            # 读取当前database subset cat 下的所有图片名(包含后缀)
            with os.scandir(image_dir) as dir:
                im_files = [file.name for file in dir if file.is_file()]
            im_files.sort()

            # 读取annotation(.mat)到Pascal3D.txt(annotation_file)
            for n in trange(
                    len(subset_list),
                    # desc=f'reading cat:{cat} for {subset} set of {database} database.'):
                    desc=f'reading {database}_{cat}_{subset}.'):
                # im_file = im_files[n]
                # im_name = os.path.splitext(im_file)[0]
                im_name = subset_list[n]
                if database == 'imagenet':
                    im_file = im_name + '.JPEG'
                else:
                    im_file = im_name + '.jpg'
                if im_file not in im_files:
                    continue
                try:
                    im_file in im_files
                except:
                    print('read img file error!')
                record = sio.loadmat(os.path.join(dataset_path, annotation_dir, im_name + '.mat'),
                                     simplify_cells=True)['record']
                objects = record['objects']
                # 如果图片中只含有一个物体,则返回字典
                # 如果含有多个物体,则返回字典的列表
                if isinstance(objects, dict):
                    objects = [objects]
                    # obj_num表示该张图片中所包含的目标数量
                obj_num = len(objects)
                # 图片中第[i]个目标记录
                for i in range(obj_num):
                    object_cls = objects[i]['class']
                    if object_cls != cat:
                        continue
                    truncated = str(objects[i]['truncated'])
                    occluded = str(objects[i]['occluded'])
                    difficult = str(objects[i]['difficult'])
                    cad_index = str(objects[i]['cad_index'])
                    viewpoint = objects[i]['viewpoint']
                    try:
                        azimuth = str(viewpoint['azimuth'])
                        elevation = str(viewpoint['elevation'])
                    except:
                        azimuth = str(viewpoint['azimuth_coarse'])
                        elevation = str(viewpoint['elevation_coarse'])
                    inplane_rotation = str(viewpoint['theta'])
                    if azimuth == '0' and elevation == '0' and inplane_rotation == '0':
                        continue
                    distance = str(viewpoint['distance'])
                    try:
                        # 记录锚点
                        anchors = objects[i]['anchors']
                        has_keypoint = str(1) if len(anchors) > 0 else str(0)
                    except:
                        has_keypoint = str(0)
                    bbox = list(objects[i]['bbox'])
                    # 以左上角为原点
                    left, upper, right, lower = str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])

                    # 写入Pascal3D.txt(annotation_file)
                    with open(os.path.join(dataset_path, annotation_file), 'a') as f:
                        f.write(
                            database + ',' + subset + ',' + cat + ',' + im_name + ',' + str(i) + ',' +
                            cad_index + ',' + truncated + ',' + occluded + ',' + difficult + ',' +
                            azimuth + ',' + elevation + ',' + inplane_rotation + ',' +
                            left + ',' + upper + ',' + right + ',' + lower + ',' +
                            has_keypoint + ',' + distance + ',' +
                            os.path.join(image_dir, im_file) + '\n')
