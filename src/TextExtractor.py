import argparse
import cv2
import numpy as np
import os
import sys


class TextExtractor:
    def __init__(self, maxsize):
        self.word_border_value = 150
        self.box_thickness_words = 5
        self.char_border_value = 90
        self.box_thickness_chars = 1
        self.maxsize = maxsize
        self.init_words_chars_containers()

    def init_words_chars_containers(self):
        self.words = []
        self.characters_from_word = []

    def read_from_image(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image = img
        self.init_words_chars_containers()

    def read_from_filename(self, path_to_image):
        if not os.path.exists(path_to_image):
            sys.exit("Bad filename provided. File: " + str(path_to_image) + " does not exist!")
        self.path_to_image = path_to_image
        self.image = cv2.imread(self.path_to_image, flags=cv2.IMREAD_GRAYSCALE)
        self.init_words_chars_containers()

    def reverse_everything(self):
        self.characters_from_word = self.characters_from_word[::-1]
        for word in self.characters_from_word:
            word = reversed(word)
        self.words = reversed(self.words)

    def add_word(self, i, img):
        """
        Adds i-th word. Function supports substitution.
        """
        if i >= len(self.words):
            self.words.append(img)
            self.characters_from_word.append([])
        else:
            self.words[i] = img

    def add_char_to_ith_word(self, i, j, img):
        """
        Adds j-th letter to i-th word. Function supports substitution. Does not check
        i bounds.
        """
        if j >= len(self.characters_from_word[i]):
            self.characters_from_word[i].append(img)
        else:
            self.characters_from_word[i][j] = img

    def resize_letter(self, letter):
        img = np.full((self.maxsize[0], self.maxsize[1]), 0, np.uint8)
        print("letter shape before reshape= " + str(letter.shape))
        height, width = letter.shape
        desired_height, desired_width = img.shape
        if height > width:
            print("letter higher")
            coefficient = desired_height / height
            print("coefficient: " + str(coefficient))
            letter = cv2.resize(letter, None, fx=coefficient, fy=coefficient, interpolation=cv2.INTER_LANCZOS4)
            height, width = letter.shape
            print("letter shape after reshape= " + str(letter.shape))
            offset = int((desired_width - width) / 2)
            print("offset = " + str(offset))
            img[:, offset:offset + width] = letter
        else:
            print("letter wider")
            coefficient = desired_width / width
            print("coefficient: " + str(coefficient))
            letter = cv2.resize(letter, None, fx=coefficient, fy=coefficient, interpolation=cv2.INTER_LANCZOS4)
            height, width = letter.shape
            print("letter shape after reshape= " + str(letter.shape))
            offset = int((desired_height - height) / 2)
            print("offset = " + str(offset))
            img[offset:offset + height, :] = letter
        print("returned image shape = " + str(img.shape))
        return img

    def character_extraction(self, display_images=False, verbose=False):
        """
        TODO: include verification by average width of a character.
        Perform wordExtraction() method first.
        It then inverts the image, divides every pixel by 32, so that it's between values [0,7]
        Then it erodes it with 2x2 kernel of uint8 and scans it.
        """
        for idxWord, word, in enumerate(self.words):
            img = word.copy()

            _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            img_copy = img.copy()

            img = np.floor(np.divide(img, 32))  # divided to get data into [0,7]
            # kernel = np.ones((2,2),np.uint8)

            # img = cv2.erode(img, kernel)
            # img_copy = cv2.erode(img_copy, kernel)

            # img_copy = cv2.bitwise_not(img_copy)
            if verbose:
                print("img")
                for row in img:
                    for cell in row:
                        sys.stdout.write("%d" % cell)
                    sys.stdout.write("\n")

            is_scanning_letter, zero_result = False, False
            upper_border, lower_border, margin = 0, 0, 2

            # detect the height of the words start
            for row in range(img.shape[0]):
                zero_result = np.isclose(np.dot(np.ones((1, img.shape[1])), img[row, :]), 0)
                if not zero_result and not is_scanning_letter:
                    upper_border = row
                    if row > margin:  # add margin to make rectangle bigger than the letter
                        upper_border -= margin
                    is_scanning_letter = True
                if is_scanning_letter and zero_result:
                    lower_border = row
                    if row < img.shape[0] - margin:
                        lower_border += margin
                    is_scanning_letter = False
                    break

            x1, x2, letters = 0, 0, ()
            avg_width, width = 100, []
            # detect breaks between letters
            for col in range(img.shape[1]):
                zero_result = np.isclose(np.dot(np.ones((1, img.shape[0])), img[:, col]), 0)
                if not zero_result and not is_scanning_letter:  # letter start
                    x1 = col
                    if x1 > margin:
                        x1 -= margin
                    is_scanning_letter = True
                    # print("poczatek literki")
                if zero_result and is_scanning_letter:  # letter end
                    width.append(x2 - x1)
                    x2 = col
                    if col < img.shape[1] - margin:
                        x2 += margin
                    is_scanning_letter = False
                    # print("koniec literki")
                    # print("szerokosc: %d" % (x2-x1))

                    letters = letters + (img_copy[upper_border:lower_border, x1:x2],)
                    cv2.rectangle(word, (x1, upper_border), (x2, lower_border), self.char_border_value,
                                  self.box_thickness_chars)

            for idxChar, letter in enumerate(letters):
                letter = self.resize_letter(letter)
                _, letter = cv2.threshold(letter, 150, 255, cv2.THRESH_BINARY)
                if display_images:
                    cv2.imshow("cropped_" + str(idxChar), letter)
                    cv2.moveWindow("cropped_" + str(idxChar), 30 * (1 + idxChar), 30 * (1 + idxChar))
                if verbose:
                    print("Letter " + str(idxChar) + " in word " + str(idxWord))
                    for row in letter:
                        for cell in row:
                            sys.stdout.write("%d" % cell)
                        sys.stdout.write("\n")
                self.add_char_to_ith_word(idxWord, idxChar, letter)
            if display_images:
                cv2.imshow("whole_word_copy_not_save_in_the_object", word)
                cv2.waitKey()
                cv2.destroyAllWindows()

    def word_extraction(self, max_h=100, max_w=200, min_h=40, min_w=40, display_images=False):
        """
        Expects grayscale image. The function is looking for contours in an image.
        https://stackoverflow.com/questions/23506105/extracting-text-opencv

        answered May 9 '14 at 5:07 by anana
        accessed Oct 10 '17 at 18:46
        """
        image = self.image.copy()
        _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
        image_to_extract = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=13)
        _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self._contours = ()
        for idx, contour in enumerate(contours):
            [x, y, w, h] = cv2.boundingRect(contour)
            # discard areas that are too large
            if h > max_h and w > max_w:
                continue
            # discard areas that are too small
            if h < min_h or w < min_w:
                continue
            cv2.rectangle(self.image, (x, y), (x + w, y + h), self.word_border_value, self.box_thickness_words)
            self._contours = self._contours + (image_to_extract[y:y + h, x:x + w],)

        for idx, img in enumerate(self._contours):
            if display_images:
                cv2.imshow("boundingRectangle_" + str(idx), img)
                cv2.moveWindow("boundingRectangle_" + str(idx), 1400, (1000 - 50 * idx))
            self.add_word(idx, img)

        # single word case
        if len(self.words) is 0:
            self.add_word(0, image_to_extract)
            if display_images:
                cv2.imshow("boundingRectangle_0", image_to_extract)

        if display_images:
            cv2.imshow("whole_text", self.image)
            cv2.waitKey()
            cv2.destroyAllWindows()


def main(single_word, multiple_words):
    print("OpenCV Version: {}".format(cv2.__version__))
    t = TextExtractor(maxsize=(64, 64))
    t.read_from_filename(multiple_words)
    t.word_extraction(100, 200, 40, 40)  # original values 300, 300, 40, 40
    t.character_extraction()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_single_word_image", help="Path to single word image")
    parser.add_argument("path_to_multiple_words_image", help="Path to multiple words image")
    args = parser.parse_args()
    main(args.path_to_single_word_image, args.path_to_multiple_words_image)
