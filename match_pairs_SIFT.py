import os
import imshowpair
import skimage
from PIL import Image
from torchvision import transforms

import numpy as np
import matplotlib.cm as cm
import torch
import sys
import cv2

import matplotlib.pyplot as plt
from SuperGlue.models.utils import make_matching_plot_fast

torch.set_grad_enabled(False)

def blend(a, b, alpha=0.5):
    """
    Alpha blend two images.
    Parameters
    ----------
    a, b : numpy.ndarray
        Images to blend.
    alpha : float
        Blending factor.
    Returns
    -------
    result : numpy.ndarray
        Blended image.
    """

    a = skimage.img_as_float(a)
    b = skimage.img_as_float(b)
    return a*alpha+(1-alpha)*b

def change_perspective_mat(H, raw_res, new_res):
    # 投影变换矩阵H是在特定分辨率下计算出来的，在新分辨率下，H需要做一定的变换
    # 这里默认原始图像和新图像分辨率都是正方形
    k = new_res / raw_res
    H[2, 2] *= k
    H[[0, 1], :] *= k
    H[[0, 1], 2] *= k
    return H

if __name__ == '__main__':

    dir_path = './'
    query_path = os.path.join(dir_path, '2.jpg')
    refer_path = os.path.join(dir_path, '1.jpg')

    query = cv2.imread(query_path, 0)
    query_rgb = cv2.imread(query_path, 1)
    refer = cv2.imread(refer_path, 0)
    refer_rgb = cv2.imread(refer_path, 1)

    assert query.shape == refer.shape
    raw_h, raw_w = query.shape[:2]

    SIFT = cv2.SIFT_create()
    kpts0, kpts1 = SIFT.detect(query), SIFT.detect(refer)
    kpts0, des0 = SIFT.compute(query, kpts0)
    kpts1, des1 = SIFT.compute(refer, kpts1)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    goodMatch = []
    matches = bf.knnMatch(des0, des1, k=2)
    for mm, nn in matches:
        #     print(m.distance, n.distance)
        if mm.distance < 0.8 * nn.distance:
            goodMatch.append(mm)

    src_pts = [kpts0[mm.queryIdx].pt for mm in goodMatch]
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = [kpts1[mm.trainIdx].pt for mm in goodMatch]
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    mconf = np.array([np.log(1/m.distance) for m in goodMatch])
    mconf = (mconf-mconf.min())/(mconf.max()-mconf.min() + 1e-6)
    M = None
    if len(src_pts) >= 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    if M is None:
        print('Registration failed')
    else:
        align_rgb = cv2.warpPerspective(query_rgb, M, (raw_w, raw_h), borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        color = cm.jet(mconf)
        show1 = make_matching_plot_fast(
            query_rgb, refer_rgb, src_pts.squeeze(), dst_pts.squeeze(), src_pts.squeeze(), dst_pts.squeeze(), color,
            'SIFT')

        ####### 1
        plt.figure(dpi=300)
        imshowpair.imshowpair(align_rgb, refer_rgb, blend)
        plt.show()

        plt.figure(dpi=300)
        plt.imshow(show1)
        plt.show()

        plt.close()
