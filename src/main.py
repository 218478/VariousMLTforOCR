from readChars74K import Reader_Chars74K, modelCNN

import argparse, logging, os, cv2

PATH_TO_TRAINED_MODEL_FROM_GUI = "/home/kkuczaj/Praca_inzynierska/build/trained_model.h5"
PATH_TO_LETTER_M = "/home/kkuczaj/Praca_inzynierska/build/cropped_0.jpg"
PATH_TO_LETTER_o = "/home/kkuczaj/Praca_inzynierska/build/cropped_1.jpg"
PATH_TO_LETTER_d = "/home/kkuczaj/Praca_inzynierska/build/cropped_2.jpg"
PATH_TO_LETTER_e = "/home/kkuczaj/Praca_inzynierska/build/cropped_3.jpg"
PATH_TO_LETTER_l = "/home/kkuczaj/Praca_inzynierska/build/cropped_4.jpg"
PATH_TO_LETTER_s = "/home/kkuczaj/Praca_inzynierska/build/cropped_5.jpg"

def main(filepath):
    classNo = 62
    maxsize = (16, 16)
    r = Reader_Chars74K(filepath, classNo)
    m = modelCNN(maxsize, classNo, PATH_TO_TRAINED_MODEL_FROM_GUI)
    img_M = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_M),cv2.COLOR_BGR2GRAY) # grayscale image read
    img_o = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_o),cv2.COLOR_BGR2GRAY) # grayscale image read
    img_d = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_d),cv2.COLOR_BGR2GRAY) # grayscale image read
    img_e = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_e),cv2.COLOR_BGR2GRAY) # grayscale image read
    img_l = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_l),cv2.COLOR_BGR2GRAY) # grayscale image read
    img_s = cv2.cvtColor(cv2.imread(PATH_TO_LETTER_s),cv2.COLOR_BGR2GRAY) # grayscale image read
    print((m.predict(img_M)))
    print((r.readableLabels[m.predict(img_M)]))
    print((m.predict(img_o)))
    print((r.readableLabels[m.predict(img_o)]))
    print((m.predict(img_d)))
    print((r.readableLabels[m.predict(img_d)]))
    print((m.predict(img_e)))
    print((r.readableLabels[m.predict(img_e)]))
    print((m.predict(img_l)))
    print((r.readableLabels[m.predict(img_l)]))
    print((m.predict(img_s)))
    print((r.readableLabels[m.predict(img_s)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pathToDatasets", help="Directory to stored datasets")
    parser.add_argument("pathToLogFileDir", help="Path to log file")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s \t %(levelname)s:%(message)s', filename=os.path.join(args.pathToLogFileDir, 'logFile.log'))#, level = logging.INFO)
    main(args.pathToDatasets)