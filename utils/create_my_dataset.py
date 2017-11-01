from PIL import Image, ImageDraw, ImageFont
from imgaug import augmenters as iaa
import os, cv2, sys
import numpy as np


"""
root
 |
 |____Sample001
 |          |
 |          |_img001-00001.png
 |          |_img001-00002.png
 |
 |____Sample002
 |      ...
"""

class DatasetCreator():
    def __init__(self, height = 16, width = 16, classNo=62):
        self.height = height
        self.width = width
        self.classNo = 62
        self.createReadableLabels()

    def _create_blank(self, width, height, rgb_color=(0, 0, 0)):
        """
        https://stackoverflow.com/questions/9710520/opencv-createimage-function-isnt-working
        Answered on 07.04.2017 1:46     Accessed on 02.11.2017 23:16
        """
        image = np.zeros((height, width, 3), np.uint8)
        color = tuple(reversed(rgb_color)) # Since OpenCV uses BGR, convert the color first
        image[:] = color # Fill image with color
        return image

    def createReadableLabels(self):
        self.readableLabels = [[]]*self.classNo
        for i in range(0,10):
            self.readableLabels[i] = str(i)
        for i in range(65,91):
            self.readableLabels[i-55] =  chr(i)
        for i in range(97,123):
            self.readableLabels[i-61] =  chr(i)

    def generateDataset(self, rootPath, fontscale = 1, count = 1):
        seq = iaa.Sequential([
                iaa.Crop(px=(0, 1)), # crop images from each side by 0 to 16px (randomly chosen)
                iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
                ])
        for c in range(self.classNo):
            suffix = "%03d" % (c+1)
            dir_name = os.path.join(rootPath, "Sample" + suffix)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(count):
                # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
                # or a list of 3D numpy arrays, each having shape (height, width, channels).
                # Grayscale images must have shape (height, width, 1) each.
                # All images must have numpy's dtype uint8. Values are expected to be in
                # range 0-255.
                images = self._putCvText(self.readableLabels[c], fontscale=fontscale, howManyInstances=count)
                # images_aug = seq.augment_images(images)
                for idx, img_aug in enumerate(images):
                    filename = "image" + suffix + "%05d.png"% idx
                    # cv2.imwrite(os.path.join(dir_name, filename),img_aug)

    def _putCvText(self, txt, fontscale, howManyInstances=1016, bckgColor = (255, 255, 255), fontColor = (0, 0, 0)):
        image = self._create_blank(self.width, self.height, rgb_color=bckgColor)
        font = cv2.FONT_HERSHEY_COMPLEX #  normal sans serif
        cv2.putText(image, str(txt), (int(self.height/2), int(self.width/2)), font, fontscale, fontColor)
        images = np.zeros((howManyInstances,self.height, self.width, 3))
        for i in range(howManyInstances): # TODO: I bet this can be nicely optimized
            images[i] = image

        print(image)
        return images

def main():
    d = DatasetCreator(16,16)
    d.generateDataset("dataset")

if __name__ == '__main__':
    main()
