# Import required modules
from api.ReconAnom import JSRNet_api
from api.SuperPoint import SuperPoint_api
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class first_info():
    def __init__(self):
        # Instantiate the JSRNet and SuperPoint APIs
        self.jsr = JSRNet_api()
        self.superpoint = SuperPoint_api()

    def extract_one_image(self, img_path, name='0', size=(768, 768)):
        # Read the image in color (1 in cv2.imread() stands for color)
        query = cv2.imread(img_path, 1)

        # Resize the image to the specified size
        query = cv2.resize(query, size)

        # Prepare inputs for the SuperPoint and JSRNet
        # Convert the query image to grayscale for SuperPoint and normalize
        sup_inp = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY) / 255.
        # Use the color image for JSRNet
        jsr_inp = query

        # Run the APIs on the inputs
        jsr_res = self.jsr.run(jsr_inp)
        sup_res = self.superpoint.run_SuperPoint(sup_inp, name)

        # Return the results from both APIs
        return jsr_res, sup_res
