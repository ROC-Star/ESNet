"""
@FileName:TE_test.py
@Author:ROC
@Time:2023/8/19 22:25
"""
import os
import time
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.dataset import test_dataset as EvalDataset
from lib.ESNet import ESNet


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()
    total = []
    with torch.no_grad():
        # test_time = AverageMeter()
        for i in range(val_loader.size):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()
            # torch.cuda.synchronize()
            begin = time.time()
            _, _, res, e = model(image)
            # torch.cuda.synchronize()
            end = time.time()
            t = end - begin
            total.append(t)
            output = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            pt = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=False)
            pt = pt.sigmoid().data.cpu().numpy().squeeze()
            pt = (pt - pt.min()) / (pt.max() - pt.min() + 1e-8)

            cv2.imwrite(map_save_path + name, output * 255)
            print('>>> saving prediction at: {}'.format(map_save_path + name))
        fps = (val_loader.size) / np.sum(total)
        print("fps:", fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snap_path', type=str,
                        default='/media/omnisky/data/rp/COD_TILNet/lib_pytorch/snapshot/Exp20/Net_epoch_150.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='train use gpu')
    parser.add_argument('--test_size', type=int, default=384,
                        help='training dataset size')
    opt = parser.parse_args()

    txt_save_path = './result6/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True
    model = ESNet().cuda()
    model.load_state_dict(torch.load(opt.snap_path))
    model.eval()
    # 'CAMO', 'COD10K', 'NC4K', 'CHAMELEON'
    for data_name in ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='/media/omnisky/data/Datasets/COD/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=opt.test_size,
        )
