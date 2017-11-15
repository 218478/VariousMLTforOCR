import cv2, argparse, os, numpy as np

def generate_img(text):
    fontColor = (0, 0, 0)
    fonts = [cv2.FONT_HERSHEY_SIMPLEX,
             cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX,
             cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_COMPLEX_SMALL]

    font = fonts[1]
    img = np.full((128,len(text)*70), 255, np.uint8)
    fontscale=4
    textsize = cv2.getTextSize(text, font, fontscale,2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)
    textY = int((img.shape[0] + textsize[1]) / 2)
    cv2.putText(img, str(text), (textX, textY ), font, fontscale, fontColor, thickness=5)
    filename = "generated.jpg"
    cv2.imwrite(filename, img)
    print("generated image in " + os.path.join(os.getcwd(), filename))

def ask_for_text():
    print("what shall I generate?")
    return input()

if __name__ == '__main__':
    generate_img(ask_for_text())