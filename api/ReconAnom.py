from jsr_code.ReconAnom import get_model
import sys
import cv2
import os
from PIL import Image
from torchvision import transforms


class JSRNet_api():
    def __init__(self):
        self.model = get_model()

    def run(self, image):
        inp0 = Image.fromarray(image)
        # inp0 = transforms.ToTensor()(inp0).unsqueeze(0).cuda()
        inp0 = transforms.ToTensor()(inp0).unsqueeze(0)
        return self.model.evaluate(inp0)

if '__name__' == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model = get_model()

