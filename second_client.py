# Import necessary libraries
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

# Function to blend two images together with a specified blending factor
def blend(a, b, alpha=0.5):
    a = skimage.img_as_float(a)
    b = skimage.img_as_float(b)
    return a*alpha+(1-alpha)*b

class second_info():
    def __init__(self, weight='outdoor', match_threshold=0.2):
        # Set SuperGlue parameters
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
        # Initialize SuperGlue model with given configuration
        self.superglue = SuperGlue(config.get('superglue', {})).eval()
        self.superpoint = SuperPoint_api()

    # Perform matching of keypoints between two images
    def matching(self, sup_data_first, img2_path, size=(768, 768)):
        refer = cv2.imread(img2_path, 1)
        refer = cv2.resize(refer, size)
        sup_inp = cv2.cvtColor(refer, cv2.COLOR_BGR2GRAY) / 255.
        sup_res_refer = self.superpoint.run_SuperPoint(sup_inp, name='1')
        data = {**sup_data_first, **sup_res_refer}
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        pred = {**data, **self.superglue(data)}
        return pred

    # Register keypoints between two images and return the transformed image
    def registration(self, pred, jsr_res=None, show_matching=True):
        # Converting tensors to numpy arrays for further processing
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Select the keypoints with valid matches
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Filter the matched keypoints based on match threshold
        match_threshold = self.match_threshold
        mkpts0 = mkpts0[mconf > match_threshold]
        mkpts1 = mkpts1[mconf > match_threshold]

        # Create an array of matched points
        src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
        dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)

        # Compute the homography matrix if enough matched points are present
        M = None
        if len(src_pts) > 4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS, 1)

        # Check if homography
        # matrix has been computed successfully
        if M is None:
            print('Registration failed')
            return None

        # Use homography to warp perspective
        align = cv2.warpPerspective(jsr_res, M, (jsr_res.shape[1], jsr_res.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

        # Show the matches between the two images if specified
        if show_matching:
            query = pred['image0'].squeeze()
            refer = pred['image1'].squeeze()
            query = (cv2.cvtColor(query, cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)
            refer = (cv2.cvtColor(refer, cv2.COLOR_GRAY2RGB)*255).astype(np.uint8)

            # Apply homography to the first image
            align_query = cv2.warpPerspective(query, M, (jsr_res.shape[1], jsr_res.shape[0]), borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=0)

            color = cm.jet(mconf)

            # Create matching plot
            show1 = make_matching_plot_fast(
                query, refer, kpts0, kpts1, mkpts0, mkpts1, color,
                'SuperGlue')

            # Display matching plot
            plt.imshow(show1)
            plt.annotate('SuperGlue Results', (0, 0), color='red', fontsize=20)
            plt.show()

            # Save matching plot
            io.savemat('SuperGlueResults.mat', {'show1': show1})

            # Display original and transformed first image, and the reference image
            plt.imshow(query)
            plt.annotate('First Gray', (0, 0), color='red', fontsize=20)
            plt.show()

            show_align = blend(align_query, refer)
            qtensor = torch.from_numpy(align_query).permute(2, 0, 1)[None, ...] / 255.
            rtensor = torch.from_numpy(refer).permute(2, 0, 1)[None, ...] / 255.

            plt.imshow(show_align)
            plt.annotate('Matching Results', (0, 0), color='red', fontsize=20)
            plt.show()

            # Save matching results
            io.savemat('MatchingResults.mat', {'show_align': show_align})

            plt.imshow(align_query)
            plt.annotate('align_query', (0, 0), color='red', fontsize=20)
            plt.show()

            # Save warped first image
            io.savemat('align_query.mat', {'align_query': align_query})

            plt.imshow(refer)
            plt.annotate('refer', (0, 0), color='red', fontsize=20)
            plt.show()

            # Save reference image
            io.savemat('refer.mat', {'refer': refer})

        return align

