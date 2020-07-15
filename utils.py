import scipy.io as sio
import Path
import os
import csv


def load_annotations(category: str, database: str, img_name: str):
    '''

    :param category: Path.categories[]
    :param sub_dataset: 'imagenet' or 'pascal'
    :param img_name:eg:'n02690373_16'
    :return:list:[]
    eg:
    _ = load_annotations(category=Path.categories[0], database='pascal', img_name='2008_000037')
    '''
    if database == 'imagenet':
        img_path = os.path.join(Path.images[category + '_' + database], img_name + '.JPEG')
    else:
        img_path = os.path.join(Path.images[category + '_' + database], img_name + '.jpg')
    img_annotation_path = os.path.join(Path.annotations[category + '_' + database], img_name + '.mat')
    img_annotation = sio.loadmat(img_annotation_path, simplify_cells=True)['record']['objects']
    # 如果图片中有多个待检测的物体,则返回多个字典组成的list
    if isinstance(img_annotation, list):
        return None
    bbox = img_annotation['bbox'].tolist()
    bbox = list(map(lambda x: x - 1, bbox))
    bbox = list(map(lambda x: round(x), bbox))
    viewpoints = []
    viewpoints.append(img_annotation['viewpoint']['azimuth'])
    viewpoints.append(img_annotation['viewpoint']['elevation'])
    viewpoints.append(img_annotation['viewpoint']['theta'])
    viewpoints = [round((x + 360.) % 360) for x in viewpoints]
    l = [os.path.relpath(img_path, Path.datasetpath)] + bbox + [0] + viewpoints

    return l


def load_train_set_index():
    '''

    :return:返回一个列表:eg:
    00000 = {str} 'n02690373_10001'
    00001 = {str} 'n02690373_10032'
    00002 = {str} 'n02690373_10061'
    00003 = {str} 'n02690373_101'
    '''
    # 读取所有种类的*imagenet_train.txt到train_set_index中
    # 11045个数据
    train_set_index = {}
    for category in Path.categories:
        with open(Path.image_sets[category + '_imagenet'] + '_train.txt', 'r') as f:
            train_set_index[str(category) + '_imagenet'] = []
            for line in f.readlines():  # 读取每行
                # train_set_index.append(line.strip())  # 去除所有行后的'\n'
                train_set_index[str(category) + '_imagenet'].append(line.strip())

    # 读取所有种类的pascal数据到train_set_index中
    # 8457个数据
    for category in Path.categories:
        train_set_index[str(category) + '_pascal'] = []
        category_paths = os.listdir(Path.images[category + '_pascal'])
        category_paths = list(map(lambda l: l[:-4], category_paths))  # 消除'.txt'
        # train_set_index += category_paths
        train_set_index[str(category) + '_pascal'] = category_paths
    return train_set_index


def load_val_set_index():
    val_set_index = {}
    # 读取所有种类的*imagenet_train.txt到val_set_index中
    # 10812个数据

    for category in Path.categories:
        with open(Path.image_sets[category + '_imagenet'] + '_val.txt', 'r') as f:
            val_set_index[str(category) + '_imagenet'] = []
            for line in f.readlines():  # 读取每行
                # val_set_index.append(line.strip())  # 去除所有行后的'\n'
                val_set_index[str(category) + '_imagenet'].append(line.strip())
    return val_set_index


train_sets = load_train_set_index()  # 19502
val_sets = load_val_set_index()  # 10812

headers = ['imgPath', 'bboxTLX', 'bboxTLY', 'bboxBRX', 'bboxBRY', 'imgKeyptX', 'imgKeyptY', 'keyptClass', 'objClass',
           'azimuthClass', 'elevationClass', 'rotationClass']


def gen_csv():
    l_train = []
    l_val = []
    for category in Path.categories:
        for database in ['imagenet', 'pascal']:
            for img_path in train_sets[category + '_' + database]:
                l_train.append(load_annotations(category=category, database=database, img_name=img_path))
            if database == 'imagenet':
                for img_path in val_sets[category + '_' + database]:
                    l_val.append(load_annotations(category=category, database=database, img_name=img_path))
    l_train = list(filter(None, l_train))  # 过滤l中所有的None
    l_val = list(filter(None, l_val))
    with open("train.csv", "w", newline='') as f:
        # newline参数控制行之间是否空行
        f_csv = csv.writer(f)
        f_csv.writerow(headers)  # headers为表头属性名组成的数组
        f_csv.writerows(l_train)
    with open("val.csv", "w", newline='') as f:
        # newline参数控制行之间是否空行
        f_csv = csv.writer(f)
        f_csv.writerow(headers)  # headers为表头属性名组成的数组
        f_csv.writerows(l_val)


