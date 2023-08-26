import cv2
from PIL import Image
import os
import pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt





def xywh2xyxy(bboxes):
    #bboxes[:, 0] = bboxes[:, 0] #- bboxes[:, 2]/2
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    #bboxes[:, 1] = bboxes[:, 1] #- bboxes[:, 3]/2
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    return bboxes

def calculate_giou(bbox_p, bbox_g):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    #batch_size = bbox_p.shape[0]
    #bbox_g = bbox_g.reshape((-1,4))
    #bbox_p = bbox_p.view(-1,4)
    #bbox_p = bbox_p.cpu().detach().numpy()
    #bbox_g = bbox_g.cpu().detach().numpy()
    x1p = np.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = np.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = np.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = np.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)

    bbox_p = np.concatenate([x1p, y1p, x2p, y2p], axis=1)
    # calc area of Bg
    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    # calc area of Bp
    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

    # cal intersection
    x1I = np.maximum(bbox_p[:, 0], bbox_g[:, 0])
    y1I = np.maximum(bbox_p[:, 1], bbox_g[:, 1])
    x2I = np.minimum(bbox_p[:, 2], bbox_g[:, 2])
    y2I = np.minimum(bbox_p[:, 3], bbox_g[:, 3])
    I = np.maximum((y2I - y1I), 0) * np.maximum((x2I - x1I), 0)

    # find enclosing box
    x1C = np.minimum(bbox_p[:, 0], bbox_g[:, 0])
    y1C = np.minimum(bbox_p[:, 1], bbox_g[:, 1])
    x2C = np.maximum(bbox_p[:, 2], bbox_g[:, 2])
    y2C = np.maximum(bbox_p[:, 3], bbox_g[:, 3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    iou = 1.0 * I / U

    # Giou
    giou = iou - (area_c - U) / area_c

    return  giou


def bbox_gious(boxes, box):
    return calculate_giou(box, boxes)


def image_preprocess(img, type_platform = "tg"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.max(img.shape) > 2048:
        k = np.max(img.shape) / 2048
        img = cv2.resize(img, (int(img.shape[1] / k), int(img.shape[0] / k)))
    if type_platform == "tg":
        return cv2.erode(img, (5, 5), 1)


def equal(word, target):
    if len(word) != len(target):
        return False

    count = 0
    for i in range(len(word)):
        if word[i] == target[i]:
            count += 1

    if abs(count - len(word)) <= 1:
        return True

    return False

def is_target_word(word, type_platform = "tg"):
    if type_platform == "vk":
        #if equal("подписчики", word):
        #    return True
        list_target = ["подписчика", "участника", "подписчиков", "участников", "участник", "подписчик", "участники"]

        for target in list_target:
            if word == target:
                return True

    if type_platform == "tg":
        list_target = ["подписчики", "подписчиков", "подписчик"]

    return False




def get_word_boxes(img, visualize = False, type_platform = "vk"):

    d = pytesseract.image_to_data(img, output_type=Output.DICT, lang= 'rus')
    n_boxes = len(d['level'])

    dict_words = {}
    dict_words["box"] = []
    dict_words["text"] = []

    box = None
    for i in range(n_boxes):
        #print("подписчик" in d['text'][i])
        #print(d['text'][i])
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        word = d['text'][i].lower()
        dict_words["box"].append([x, y, w, h])
        dict_words["text"].append(word)
        '''if visualize:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
'''

        if is_target_word(word, type_platform) and box is None:
            box = [x, y, w, h]

        #if "подписчика" in d['text'][i].lower() and box is None:
        #    box = [x, y, w, h]








    return dict_words, box




#def find_near_boxes():
#    bbox_gious

def is_digit(word):
    flag = False
    for c in word:
        if c >= "0" and c <= "9":
            flag = True
        else:
            flag = False
            break
    return flag

def find_digits(dict_words, box, thresh = 0.8, type_platform = "vk"):
    #print(box, "\n")
    #print(dict_words['box'])


    #print(np.round(gious, 1))

    if thresh < 1.0:
        gious = bbox_gious(np.array(dict_words['box']), np.array([box]))
        gious = 1 - (gious + 1) / 2
        sorted_gious = np.sort(gious)
        sorted_idxs = np.argsort(gious)
    else:
        sorted_gious = [0] * len(dict_words['box'])
        sorted_idxs = list(range(len(sorted_gious)))

    ##print(box)
    #print(np.array(dict_words['text'])[sorted_idxs])

    for i, giou in enumerate(sorted_gious):
        word = dict_words['text'][sorted_idxs[i]].lower().strip()
        print(word, giou)

        if giou < thresh:
            if is_digit(word):
                print("digit box", dict_words['box'][sorted_idxs[i]])
                return word
            elif "." in word and ("k" in word or "к" in word):
                word = float(word[:-1]) * 1000
                return word # convert_to_normal_digit
            elif "к" in word and is_digit(word[:-1]):
                return int(word[:-1]) * 1000


    return None


def get_count_subs(path, flag_not_find = False, type_platform = "tg"):
    img = cv2.imread(path)
    print(path)
    img_pil = Image.open(path)
    img = image_preprocess(img, type_platform)

    #print(pytesseract.image_to_string(img, lang= 'rus'))

    dict_words, box = get_word_boxes(img, type_platform)

    if box is None:
        return "Загрузите другую фотографию"

    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    box_xyxy = [x1, y1, x2, y2]
    dict_words["box"] = xywh2xyxy(np.array(dict_words["box"]))




    out = find_digits(dict_words, box_xyxy)
    #print(len(out))



    if not(out is None): # Если нашли число
        return out



    else: # Если не нашли
        print("Поиск дальше...")
        # Детектим все заново
        #img = cv2.imread(path)
        #img = image_preprocess(img, type_platform)
        #x, y, w, h = box
        x1, y1, x2, y2 = 0, y - 5 * h, img.shape[1], y + 5*h

        roi = img[y1:y2, x1:x2]
        #show_image(roi)
        #print(pytesseract.image_to_string(roi, lang= 'rus'))

        dict_words, box = get_word_boxes(roi)
        ###print(dict_words)
        out = find_digits(dict_words, box, thresh = 1.0)

    return out
