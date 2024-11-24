"""
@File: GT2Skelete.py
@Time: 2024/6/11
@Author: rp
@Software: PyCharm

"""

import os
import cv2
import numpy as np

# # 10.17 形态算法之骨架 (skimage.morphology.skeletonize)
from skimage import morphology
from tqdm import tqdm


def read_path(path, skelete_path, edge_path):
    name_list = sorted(os.listdir(path))
    for filename in tqdm(name_list, total=len(os.listdir(path))):
        # print(filename)
        imgGray = cv2.imread(path + '/' + filename, flags=0)  # flags=0 灰度图像
        edge = cv2.imread(edge_path + '/' + filename, flags=0)  # flags=0 灰度图像
        ret, imgBin = cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY)  # 二值化处理
        imgBin[imgBin == 255] = 1
        skeleton01 = morphology.skeletonize(imgBin)
        skeleton = skeleton01.astype(np.uint8) * 255

        #
        kernel = np.ones((2,2),np.uint8)
        dilated_skeleton  = cv2.dilate(skeleton,kernel,iterations=1)


        after = dilated_skeleton + np.array(edge)
        cv2.imwrite(skelete_path + '/' + filename, after)
    print('>>>>>>>>>>>>Finsh！>>>>>>>>>>>>>>>')




if __name__ == '__main__':
    ori = '/media/omnisky/data/Datasets/COD/TestDataset/COD10K/GT'
    after = '/media/omnisky/data/Datasets/COD/TrainDataset/skelete1'
    edge_path = '/media/omnisky/data/Datasets/COD/TrainDataset/Edge'
    if not os.path.exists(after):
        os.makedirs(after)

    read_path(ori, after, edge_path)
