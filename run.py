import torch
import scipy.io as io
from first_client import first_info
from second_client import second_info
import os
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

f1 = '1.jpg'
f2 = '2.jpg'
size = (526, 526)
with torch.no_grad():
    first = first_info()
    second_info = second_info()

    '''extract_one_image'''
    jsr_res, sup_res_query = first.extract_one_image(f1, size=size)
    plt.imshow(jsr_res.cpu().numpy()[0, :, :])
    plt.annotate('First', (0, 0), color='red', fontsize=20)
    plt.show()

    # save the segment result
    jnum = jsr_res.cpu().numpy()
    io.savemat('FirstDetect.mat', {'jnum': jnum})
    # matpath = 'jnum.mat'
    # io.savemat(matpath, {'jnum': jnum})

    first_save = {'jsr_res': jsr_res, **sup_res_query}
    np.save('first_info.npy', first_save)

    a = sup_res_query['keypoints0'][0]
    aa = [t.cpu().numpy() for t in a]
    io.savemat('keypoints.mat', {'aa': aa})

    b = sup_res_query['scores0'][0]
    bb = [t.cpu().numpy() for t in b]
    io.savemat('scores.mat', {'bb': bb})

    c = sup_res_query['descriptors0'][0]
    cc = [t.cpu().numpy() for t in c]
    io.savemat('descriptors.mat', {'cc': cc})


    '''从保存中读取'''
    # first_save = np.load('first_info.npy', allow_pickle=True).item()
    # jsr_res = first_save['jsr_res']
    # sup_res_query = first_save

    # '''error in wireless transmission of location'''
    # wa = a.shape
    # rana = torch.rand(wa).to(a.device)
    # p = 0.1
    # amin = a.min()
    # amax = a.max()
    # a[rana < p] = torch.rand((rana < p).sum()).to(a.device) * (amax - amin) + amin
    # sup_res_query['keypoints0'][0] = a

    # Error in wireless transmission of descriptors
    wc = c.shape
    ranc = torch.rand(wc).to(c.device)
    p = 0.01
    cmin = c.min()
    cmax = c.max()
    c[ranc < p] = torch.rand((ranc < p).sum()).to(c.device) * (cmax - cmin) + cmin
    sup_res_query['descriptors0'][0] = c

    supglue_res = second_info.matching(sup_res_query, f2, size=size)
    jsr_res = jsr_res.cpu().numpy().squeeze()
    jsr_align = second_info.registration(supglue_res, jsr_res)



plt.imshow(jsr_align)
plt.annotate('Detection Resulte of the second figure', (0, 0), color='red', fontsize=20)
plt.show()
io.savemat('SecondDetect.mat', {'jsr_align': jsr_align})
