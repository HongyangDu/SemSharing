import os
# import imshowpair
import skimage
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.cm as cm
import torch
import sys

from SuperGlue.models.matching import Matching
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


def load_superglue():
    superglue = 'outdoor'
    max_keypoints = 1024
    keypoint_threshold = 0.005
    nms_radius = 4
    sinkhorn_iterations = 20
    match_threshold = 0.2

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
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

    matching = Matching(config).eval().to(device)
    return matching, device, match_threshold


if __name__ == '__main__':

    matching, device, match_threshold = load_superglue()

    dir_path = './'
    query_path = os.path.join(dir_path, '2.jpg')
    refer_path = os.path.join(dir_path, '1.jpg')

    query = cv2.imread(query_path, 0) / 255.
    query_rgb = cv2.imread(query_path, 1)
    refer = cv2.imread(refer_path, 0) / 255.
    refer_rgb = cv2.imread(refer_path, 1)

    query_rgb = cv2.resize(query_rgb, (768, 768))
    refer_rgb = cv2.resize(refer_rgb, (768, 768))
    query = cv2.resize(query, (768, 768))
    refer = cv2.resize(refer, (768, 768))

    assert query.shape == refer.shape
    raw_h, raw_w = query.shape[:2]

    inp0 = Image.fromarray(query)
    inp0 = transforms.ToTensor()(inp0).unsqueeze(0).to(device)
    inp1 = Image.fromarray(refer)
    inp1 = transforms.ToTensor()(inp1).unsqueeze(0).to(device)

    # Perform the matching.
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    mkpts0 = mkpts0[mconf > match_threshold]
    mkpts1 = mkpts1[mconf > match_threshold]

    # Keep the matching keypoints to show.

    cv_kpts1 = [cv2.KeyPoint(int(i[0]), int(i[1]), 20)
                for i in mkpts0]
    cv_kpts2 = [cv2.KeyPoint(int(i[0]), int(i[1]), 20)
                for i in mkpts1]

    src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
    dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)

    M = None
    if len(src_pts) > 4:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 1)

    if M is None:
        print('Registration failed')
    else:
        align_rgb = cv2.warpPerspective(query_rgb, M, (raw_w, raw_h), borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
        color = cm.jet(mconf)
        show1 = make_matching_plot_fast(
            query_rgb, refer_rgb, kpts0, kpts1, mkpts0, mkpts1, color,
            'SuperGlue')

        ####### 1
        plt.figure(dpi=300)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        # imshowpair.imshowpair(align_rgb, refer_rgb, blend)
        plt.show()

        plt.figure(dpi=300)
        plt.imshow(show1)
        plt.show()

        plt.close()
