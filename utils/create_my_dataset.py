from PIL import Image, ImageDraw, ImageFont
import imgaug as ia
from imgaug import augmenters as iaa
import os, cv2, sys, time
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
        self.classNo = classNo
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

    def commonAugmentation(self):
        '''
        iaa.Sequential([
                iaa.Crop(px=(0, 1)), # crop images from each side by 0 to 16px (randomly chosen)
                iaa.GaussianBlur(sigma=(0, 1.0)), # blur images with a sigma of 0 to 3.0
                iaa.Sharpen(alpha=1.0)
                ], random_order=True)

        http://imgaug.readthedocs.io/en/latest/source/examples_basics.html
        '''
        ia.seed(1)
        seq = iaa.Sequential([
            iaa.Crop(percent=(0, 0.1)), # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.3))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.875, 1.125)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.9, 1.1), per_channel=0.1),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-2.5, 2.5),
                shear=(-8, 8)
            )
        ], random_order=True) # apply augmenters in random order
        return seq


    def generateDataset(self, rootPath, fontscale = 0.5, countForClass = 1016):
        seq = self.commonAugmentation()
        sys.stdout.write("[%s]" % (" " * self.classNo))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.classNo+1)) # return to start of line, after '['
        for c in range(self.classNo):
            suffix = "%03d" % (c+1)
            dir_name = os.path.join(rootPath, "Sample" + suffix)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            for i in range(countForClass):
                # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
                # or a list of 3D numpy arrays, each having shape (height, width, channels).
                # Grayscale images must have shape (height, width, 1) each.
                # All images must have numpy's dtype uint8. Values are expected to be in
                # range 0-255.
                images = self._putCvText(self.readableLabels[c], fontscale=fontscale, howManyInstances=countForClass)
                images_aug = seq.augment_images(images)
                for idx, img_aug in enumerate(images_aug):
                    filename = "image" + suffix + "%05d.png"% idx
                    cv2.imwrite(os.path.join(dir_name, filename),img_aug)
            sys.stdout.write("-")
            sys.stdout.flush()

        sys.stdout.write("\n")

    def _putCvText(self, txt, fontscale, howManyInstances=1016, bckgColor = (255, 255, 255), fontColor = (0, 0, 0)):
        image = self._create_blank(self.width, self.height, rgb_color=bckgColor)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(txt), (int(self.height/4), int(self.width/5*4)), font, fontscale, fontColor)
        images = np.zeros((howManyInstances,self.height, self.width, 3))
        for i in range(howManyInstances): # TODO: I bet this can be nicely optimized
            images[i] = image
        return images

def main():
    start = time.clock()
    d = DatasetCreator(height=16, width=16, classNo=62)
    d.generateDataset("dataset",fontscale = 0.5, countForClass=10)
    stop = time.clock()
    print("It took me %f seconds" % (stop-start))
if __name__ == '__main__':
    main()
