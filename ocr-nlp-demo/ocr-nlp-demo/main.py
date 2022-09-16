from pprint import pprint

from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from paddlenlp import Taskflow

def quick_test():
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
    img_path = "./test_img/hetong2.jpg"
    result = ocr.ocr(img_path, cls=False)
    for line in result:
        print(line)



    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.show()
    im_show.save("./test_img/ocr_hetong2.jpg")


def img_preprocess():
    # 读入图像,三通道
    image = cv2.imread("./test_img/hetong2.jpg", cv2.IMREAD_COLOR)  # timg.jpeg

    # 获得三个通道
    Bch, Gch, Rch = cv2.split(image)

    # 保存三通道图片
    cv2.imwrite('blue_channel.jpg', Bch)
    cv2.imwrite('green_channel.jpg', Gch)
    cv2.imwrite('red_channel.jpg', Rch)


def ocr_extract():
    ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
    img_path = './red_channel.jpg'
    result = ocr.ocr(img_path, cls=False)

    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
    im_show = Image.fromarray(im_show)
    vis = np.array(im_show)
    im_show.show()
    im_show.save("./ocr_red_channel.jpg")

    all_context = "\n".join(txts)
    print(all_context)
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(all_context)


def information_extraction():
    schema = ["甲方", "乙方", "总价"]
    ie = Taskflow('information_extraction', schema=schema)
    ie.set_schema(schema)
    all_context = ''

    with open('result.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            all_context += line
    pprint(ie(all_context))

if __name__ == '__main__':
    quick_test()
    img_preprocess()
    ocr_extract()
    information_extraction()
