import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import os, cv2, sys, time, multiprocessing
from joblib import Parallel, delayed


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
    '''
    Parallel generator of 62 classes - 10 digits and 26 of each capital and normal letters.
    '''
    def __init__(self, height = 16, width = 16, classNo=62, bckgColor = (255, 255, 255)):
        self.height = height
        self.width = width
        self.classNo = classNo
        self._blank = self._create_blank(self.width, self.height, rgb_color=bckgColor)
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
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.2,
                iaa.GaussianBlur(sigma=(0, 0.3))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.92, 1.08)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.1),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.9, 1.1), per_channel=0.1),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                rotate=(-1.5, 1.5),
                cval=255
            )
        ], random_order=True) # apply augmenters in random order
        return seq

    def generateImagesForClass(self, c, rootPath, fontscale, countForClass, seq):
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
            fonts = [cv2.FONT_HERSHEY_SIMPLEX,
                     cv2.FONT_HERSHEY_DUPLEX,
                     cv2.FONT_HERSHEY_COMPLEX,
                     cv2.FONT_HERSHEY_TRIPLEX,
                     cv2.FONT_HERSHEY_COMPLEX_SMALL]
            for j, font in enumerate(fonts):
                images = self._putCvText(self.readableLabels[c], font=font, fontscale=fontscale, howManyInstances=countForClass)
                images_aug = seq.augment_images(images)
                for idx, img_aug in enumerate(images_aug):
                    filename = "image" + suffix + "%05d.png"% (idx+j*countForClass)
                    cv2.imwrite(os.path.join(dir_name, filename),img_aug)
        sys.stdout.write("-")
        sys.stdout.flush()

    def generateDataset(self, rootPath, fontscale = 0.5, countForClass = 1016):
        seq = self.commonAugmentation()
        sys.stdout.write("[%s]" % (" " * self.classNo))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.classNo+1)) # return to start of line, after '['
        inputs = range(self.classNo)
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.generateImagesForClass)(i, rootPath, fontscale, countForClass, seq) for i in inputs)
        sys.stdout.write("\n")

    def _putCvText(self, text, font, fontscale, howManyInstances=1016, fontColor = (0, 0, 0)):
        """
        This function can be randomized to choose different font every time ;)

        Centering of the text thanks to https://gist.github.com/xcsrz/8938a5d4a47976c745407fe2788c813a
        accessed on 05.11.2017 23:43        posted on 08.03.2017
        """
        img = np.full((128,128), 255, np.uint8)
        fontscale=4
        textsize = cv2.getTextSize(text, font, fontscale,2)[0]

        # get coords based on boundary
        textX = int((img.shape[1] - textsize[0]) / 2)
        textY = int((img.shape[0] + textsize[1]) / 2)

        # add text centered on image
        cv2.putText(img, str(text), (textX, textY ), font, fontscale, fontColor, thickness=2)
        res = cv2.resize(img,(16,16), interpolation = cv2.INTER_AREA)
        _,res = cv2.threshold(res,200,255,cv2.THRESH_BINARY)

        return [res]*howManyInstances

def main():
    start = time.time()
    d = DatasetCreator(height=16, width=16, classNo=62)
    d.generateDataset("dataset5",fontscale = 0.5, countForClass=200)
    stop = time.time()
    print("It took me %f seconds" % (stop-start))
if __name__ == '__main__':
    main()
