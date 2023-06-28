import matplotlib.pyplot as plt
import scipy.io as io
from SuperGlue.models.superglue import SuperGlue
from SuperGlue.models.utils import make_matching_plot_fast
from api.ReconAnom import JSRNet_api
from api.SuperPoint import SuperPoint_api
import cv2
import os
import torch
import skimage
import numpy as np
import matplotlib.cm as cm
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# from piq import ssim, SSIMLoss, haarpsi, HaarPSILoss, dss, DSSLoss


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

class second_info():
    def __init__(self, weight='outdoor', match_threshold=0.2):
        superglue = weight
        sinkhorn_iterations = 30
        match_threshold = match_threshold
        self.match_threshold = match_threshold
        config = {

            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        # self.superglue = SuperGlue(config.get('superglue', {})).eval().cuda()
        self.superglue = SuperGlue(config.get('superglue', {})).eval()
        self.superpoint = SuperPoint_api()

    def matching(self, sup_data_first, img2_path, size=(768, 768)):

        refer = cv2.imread(img2_path, 1)

        refer = cv2.resize(refer, size)

        sup_inp = cv2.cvtColor(refer, cv2.COLOR_BGR2GRAY) / 255.

        sup_res_refer = self.superpoint.run_SuperPoint(sup_inp, name='1')

        data = {**sup_data_first, **sup_res_refer}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**data, **self.superglue(data)}

        return pred

    def registration(self, pred, jsr_res=None, show_matching=True):
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        match_threshold = self.match_threshold

        mkpts0 = mkpts0[mconf > match_threshold]
        mkpts1 = mkpts1[mconf > match_threshold]

        # Keep the matching keypoints to show.

        # cv_kpts1 = [cv2.KeyPoint(int(i[0]), int(i[1]), 20)
        #             for i in mkpts0]
        # cv_kpts2 = [cv2.KeyPoint(int(i[0]), int(i[1]), 20)
        #             for i in mkpts1]

        src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
        dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)

        M = None
        if len(src_pts) > 4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 1)

        if M is None:
            print('Registration failed')
            return None
        align = cv2.warpPerspective(jsr_res, M, (jsr_res.shape[1], jsr_res.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)
        # return align
        # else:
        if show_matching:
            query = pred['image0'].squeeze()
            refer = pred['image1'].squeeze()
            query = (cv2.cvtColor(query, cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)
            refer = (cv2.cvtColor(refer, cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)
            align_query = cv2.warpPerspective(query, M, (jsr_res.shape[1], jsr_res.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)
            color = cm.jet(mconf)
            show1 = make_matching_plot_fast(
                query, refer, kpts0, kpts1, mkpts0, mkpts1, color,
                'SuperGlue')
            plt.imshow(show1)
            plt.annotate('SUerGlue Results', (0, 0), color='red', fontsize=20)
            plt.show()
            io.savemat('SUperGlueResults.mat', {'show1': show1})

            plt.imshow(query)
            plt.annotate('First GRay', (0, 0), color='red', fontsize=20)
            plt.show()

            show_align = blend(align_query, refer)
            qtensor = torch.from_numpy(align_query).permute(2, 0, 1)[None, ...] / 255.
            rtensor = torch.from_numpy(refer).permute(2, 0, 1)[None, ...] / 255.


            plt.imshow(show_align)
            plt.annotate('Matching Results', (0, 0), color='red', fontsize=20)
            plt.show()
            io.savemat('MatchingResults.mat', {'show_align': show_align})

            plt.imshow(align_query)
            plt.annotate('align_query', (0, 0), color='red', fontsize=20)
            plt.show()
            io.savemat('align_query.mat', {'align_query': align_query})

            plt.imshow(refer)
            plt.annotate('refer', (0, 0), color='red', fontsize=20)
            plt.show()
            io.savemat('refer.mat', {'refer': refer})

        return align