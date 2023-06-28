from SuperGlue.models.superpoint import SuperPoint

import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class SuperPoint_api():
    def __init__(self):
        superglue = 'outdoor'
        max_keypoints = 150
        keypoint_threshold = 0.005
        nms_radius = 4
        sinkhorn_iterations = 20
        match_threshold = 0.2

        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

        # self.superpoint = SuperPoint(config.get('superpoint', {})).eval().cuda()
        self.superpoint = SuperPoint(config.get('superpoint', {})).eval()

    def run_SuperPoint(self, query, name='0'):
        superpoint = self.superpoint

        #
        # raw_h, raw_w = query.shape[:2]

        inp0 = Image.fromarray(query)
        # inp0 = transforms.ToTensor()(inp0).unsqueeze(0).cuda()
        inp0 = transforms.ToTensor()(inp0).unsqueeze(0)
        pred = {}
        pred0 = superpoint({'image': inp0})
        pred = {**pred, **{k + name: v for k, v in pred0.items()}}
        pred = {**{'image' + name: inp0}, **pred}

        return pred


if '__name__' == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    s = SuperPoint_api()
    size = (768, 768)
    query = cv2.imread('2.jpg', 1)
    # query_rgb = cv2.imread(query_path, 1)

    query = cv2.resize(query, size) / 255.
    s.run_SuperPoint(cv2.cvtColor(query, cv2.COLOR_BGR2GRAY))
