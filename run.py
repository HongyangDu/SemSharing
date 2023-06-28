import torch
import scipy.io as io
from first_client import first_info
from second_client import second_info
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the image files and size
f1 = '1.jpg'
f2 = '2.jpg'
size = (526, 526)

with torch.no_grad():
    # Initialize the information from two clients
    first = first_info()
    second = second_info()

    # Extract information from the first image
    jsr_res, sup_res_query = first.extract_one_image(f1, size=size)

    # Show the first image
    plt.imshow(jsr_res.cpu().numpy()[0, :, :])
    plt.annotate('First', (0, 0), color='red', fontsize=20)
    plt.show()

    # Save the segment result as a .mat file
    jnum = jsr_res.cpu().numpy()
    io.savemat('FirstDetect.mat', {'jnum': jnum})

    # Save the first client's information as a .npy file
    first_save = {'jsr_res': jsr_res, **sup_res_query}
    np.save('first_info.npy', first_save)

    # Save keypoints, scores, and descriptors as .mat files
    aa = [t.cpu().numpy() for t in sup_res_query['keypoints0'][0]]
    io.savemat('keypoints.mat', {'aa': aa})

    bb = [t.cpu().numpy() for t in sup_res_query['scores0'][0]]
    io.savemat('scores.mat', {'bb': bb})

    cc = [t.cpu().numpy() for t in sup_res_query['descriptors0'][0]]
    io.savemat('descriptors.mat', {'cc': cc})

    # Error in wireless transmission of descriptors
    wc = sup_res_query['descriptors0'][0].shape
    ranc = torch.rand(wc).to(sup_res_query['descriptors0'][0].device)
    p = 0.01
    cmin = sup_res_query['descriptors0'][0].min()
    cmax = sup_res_query['descriptors0'][0].max()
    sup_res_query['descriptors0'][0][ranc < p] = torch.rand((ranc < p).sum()).to(
        sup_res_query['descriptors0'][0].device) * (cmax - cmin) + cmin

    # Match the second image
    supglue_res = second.matching(sup_res_query, f2, size=size)

    # Align the images
    jsr_res = jsr_res.cpu().numpy().squeeze()
    jsr_align = second.registration(supglue_res, jsr_res)

# Show the detection results of the second image
plt.imshow(jsr_align)
plt.annotate('Detection Resulte of the second figure', (0, 0), color='red', fontsize=20)
plt.show()

# Save the detection results of the second image
io.savemat('SecondDetect.mat', {'jsr_align': jsr_align})
