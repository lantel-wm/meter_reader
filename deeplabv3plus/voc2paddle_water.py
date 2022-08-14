import os
import shutil

data_path = '../datasets/water_deep/'
train_img_path = '../datasets/water_deep/images/train/'
val_img_path = '../datasets/water_deep/images/val/'
train_seg_path = '../datasets/water_deep/annotations/train/'
val_seg_path = '../datasets/water_deep/annotations/val/'

voc_data_path = '../datasets/water_deep_voc/'
voc_img_path = '../datasets/water_deep_voc/JPEGImages/'
voc_seg_path = '../datasets/water_deep_voc/SegmentationClass/'
voc_set_path = '../datasets/water_deep_voc/ImageSets/Segmentation/'

def split_dataset():
    with open(os.path.join(data_path, 'train.txt'), 'w') as f:
        for img_name in os.listdir(train_img_path):
            # print(img_name)
            seg_name = img_name[:-3] + 'png'

            img_file = os.path.join(train_img_path, img_name)
            seg_file = os.path.join(train_seg_path, seg_name)

            f.writelines(img_file[23:] + ' ' + seg_file[23:] + '\n')

    with open(os.path.join(data_path, 'val.txt'), 'w') as f:
        for img_name in os.listdir(val_img_path):
            # print(img_name)
            seg_name = img_name[:-3] + 'png'

            img_file = os.path.join(val_img_path, img_name)
            seg_file = os.path.join(val_seg_path, seg_name)

            f.writelines(img_file[23:] + ' ' + seg_file[23:] + '\n')

    with open(os.path.join(data_path, 'test.txt'), 'w') as f:
        pass

    with open(os.path.join(data_path, 'labels.txt'), 'w') as f:
        f.writelines('background\nwater\nmeter\n')

def copy_data(train_list, val_list):

    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)
    if not os.path.exists(train_seg_path):
        os.makedirs(train_seg_path)
    if not os.path.exists(val_img_path):
        os.makedirs(val_img_path)
    if not os.path.exists(val_seg_path):
        os.makedirs(val_seg_path)

    for train_data in train_list:
        img_name = train_data + '.jpg'
        seg_name = train_data + '.png'

        img_file = os.path.join(voc_img_path, img_name)
        new_img_file = os.path.join(train_img_path, img_name)
        shutil.copyfile(img_file, new_img_file)

        seg_file = os.path.join(voc_seg_path, seg_name)
        new_seg_file = os.path.join(train_seg_path, seg_name)
        shutil.copyfile(seg_file, new_seg_file)

    for val_data in val_list:
        img_name = val_data + '.jpg'
        seg_name = val_data + '.png'

        img_file = os.path.join(voc_img_path, img_name)
        new_img_file = os.path.join(val_img_path, img_name)
        shutil.copyfile(img_file, new_img_file)

        seg_file = os.path.join(voc_seg_path, seg_name)
        new_seg_file = os.path.join(val_seg_path, seg_name)
        shutil.copyfile(seg_file, new_seg_file)



def get_train_val():
    train_list, val_list = [], []

    with open(os.path.join(voc_set_path, 'train.txt')) as f:
        for line in f.readlines():
            train_list.append(line[:-1]) # 去掉\n

    with open(os.path.join(voc_set_path, 'val.txt')) as f:
        for line in f.readlines():
            val_list.append(line[:-1])

    return train_list, val_list


if __name__ == '__main__':
    # train_list, val_list = get_train_val()
    # copy_data(train_list, val_list)
    split_dataset()
