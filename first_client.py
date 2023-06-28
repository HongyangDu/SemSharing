from api.ReconAnom import JSRNet_api
from api.SuperPoint import SuperPoint_api
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class first_info():
    def __init__(self):
        self.jsr = JSRNet_api()
        self.superpoint = SuperPoint_api()
    def extract_one_image(self, img_path, name='0', size = (768, 768)):

        query = cv2.imread(img_path, 1)
        # query_rgb = cv2.imread(query_path, 1)

        query = cv2.resize(query, size)

        sup_inp = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY) / 255.
        jsr_inp = query

        jsr_res = self.jsr.run(jsr_inp)
        sup_res = self.superpoint.run_SuperPoint(sup_inp, name)


        return jsr_res, sup_res