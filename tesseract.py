import cv2
from PIL.Image import Image
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

IMAGE_SIZE = 1800


def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                    3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def image_preprocess(img, type_platform = "tg"):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.max(img.shape) > 3000:
        k = np.max(img.shape) / 3000
        img = cv2.resize(img, (int(img.shape[1] / k), int(img.shape[0] / k)))

    skip = False
    global BINARY_THREHOLD
    if "vk" in type_platform:
        BINARY_THREHOLD = 180
    elif type_platform == "tg":
        BINARY_THREHOLD = 160
    elif type_platform == "zn":
        BINARY_THREHOLD = 170
        #skip = True
        img = cv2.bilateralFilter(img,9,75,75)
        _, img = cv2.threshold(img, 228, 255, cv2.THRESH_BINARY)
        return img
    elif "yt" in type_platform:
        BINARY_THREHOLD = 180

    #img = cv2.dilate(img, (7, 7), 1)

    if np.mean(img) < 100:
        img = (255 - img).astype(np.uint8)

    img = remove_noise_and_smooth(img)
    #img =  cv2.erode(img, (7, 7), 3)

    return img

def equal(word, target):
    if len(word) != len(target):
        return False
    if len(word) < 5:
        return word == target





    count = 0
    for i in range(len(word)):
        if word[i] == target[i]:
            count += 1

    if abs(count - len(word)) <= 2:
        return True

    return False

def is_target_word(word, type_platform = "tg"):
    list_target = []
    if type_platform == "vk":
        #if equal("подписчики", word):
        #    return True
        list_target = ["подписчика", "подписчики", "участника", "подписчиков", "участников", "участник", "подписчик", "участники"]
        #list_target = ["подписчика", "подписчики", "подписчиков", "подписчик"]

    if type_platform == "vk_friend":
        list_target = ["друзья", "друзей"]

    if type_platform == "tg":
        list_target = ["vr", "err"]

    if type_platform == "yt1":
        list_target = ["подписчики", "подписчиков", "подписчики", "подписчика"]

    if type_platform == "yt2":
        list_target = ["просмотров", "просмотры"]

    if type_platform == "zn":
        list_target = ["дочитывания"]

    #list_target = []
    for target in list_target:
            #if word == target:

            if equal(word, target):
                return True

    return False



def find_in_line(img, type_platform):
    del_chars = ["`","/", "$", "@", "!", "©", "‘", "*", "°"]
    #del_chars = "`°"

    lines = pytesseract.image_to_string(img, lang= 'rus').split("\n")
    #print(lines)

    for line in lines:
        words = line.split()
        for i, word in enumerate(words):
            for char in del_chars:
                #print(char)
                words[i] = word.replace(char, "")
            words[i] = words[i].lower()

        for i, word in enumerate(words):

            word = word.lower()
            if is_target_word(word, type_platform):
                flag_digit_in_line = False
                digit = 0


                #print("TRUE", words)
                for p, j in enumerate(range(i-1, -1, -1)):
                    if is_digit(words[j]):
                        flag_digit_in_line = True
                        digit +=  int(words[j]) * 1000**p
                    else:
                        break

                if flag_digit_in_line:
                    return digit

                count = 0
                for p, j in enumerate(range(i+1, len(words))):

                    if is_digit(words[j]):
                        count +=1
                    else:
                        break


                for p, j in enumerate(range(i+1, len(words))):

                    if is_digit(words[j]):
                        flag_digit_in_line = True
                        digit += int(words[j]) * 1000**(count - p - 1)
                    else:
                        break

                if flag_digit_in_line:
                    return digit

    return None


def get_word_boxes(img, type_platform = "vk", visualize = False):

    del_chars = "`/,\{}:;'$#@!)?(-+°©‘"

    if type_platform == "tg":
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
    else:
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
        for char in del_chars:
            word = word.replace(char, "")


        dict_words["box"].append([x, y, w, h])
        dict_words["text"].append(word)

        if visualize:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if is_target_word(word, type_platform) and box is None:
            print(word)
            box = [x, y, w, h]

        #if "подписчика" in d['text'][i].lower() and box is None:
        #    box = [x, y, w, h]



    if visualize:
        cv2.imwrite("img_boxes.jpg", img)




    return dict_words, box




#def find_near_boxes():
#    bbox_gious

def is_digit(word):
    flag = False
    if word == ".":
        return flag

    for c in word:
        if c >= "0" and c <= "9" or c == ".":
            flag = True
        else:
            flag = False
            break
    return flag

def is_metric(word):
    flag = False
    if not("%" in word):
        return flag


    for c in word[:-1]:
        if c >= "0" and c <= "9" or c == ".":
            flag = True
        else:
            flag = False
            break

    return flag

def find_digits(dict_words, box, thresh = 0.8, type_platform = "vk"):
    #print(box, "\n")
    #print(dict_words['box'])


    #print(np.round(gious, 1))

    #if thresh < 1.0:
    gious = bbox_gious(np.array(dict_words['box']), np.array([box]))
    gious = 1 - (gious + 1) / 2
    sorted_gious = np.sort(gious)
    sorted_idxs = np.argsort(gious)
    #else:
    #    sorted_gious = [0] * len(dict_words['box'])
    #    sorted_idxs = list(range(len(sorted_gious)))

    ##print(box)
    #print(np.array(dict_words['text'])[sorted_idxs])

    for i, giou in enumerate(sorted_gious):
        word = dict_words['text'][sorted_idxs[i]].lower().strip()
        #print(word, giou, dict_words['box'][sorted_idxs[i]])

        if giou < thresh:
            if is_digit(word):
                print("digit box", dict_words['box'][sorted_idxs[i]])
                return word
            elif is_metric(word):
                return word[:-1]
            elif "." in word and ("k" in word or "к" in word) and is_digit(word.replace("к", "")):
                word = float(word[:-1]) * 1000
                return word # convert_to_normal_digit
            elif "к" in word and is_digit(word[:-1]):
                return int(word[:-1]) * 1000


    return None

def get_vk_friend(img):
    dict_words, box = get_word_boxes(img, "vk_friend", False)
    if box is None:
        return 0
    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    box_xyxy = [x1, y1, x2, y2]
    #print(box_xyxy)
    dict_words["box"] = xywh2xyxy(np.array(dict_words["box"]))




    out = find_digits(dict_words, box_xyxy)


def get_count_subs(path, type_platform = "tg"):
    img = cv2.imread(path, 0)
    img = image_preprocess(img, type_platform)

    if type_platform == "vk":
        out = find_in_line(img, type_platform)

        if  not (out is None):
            return out


    if type_platform == "zn":
        dict_words, box = get_word_boxes(img, type_platform, False)
        #print(box)
        count = 0
        for word in dict_words["text"]:
            if is_target_word(word, type_platform):
                count += 1
        print("count", count)
        if count > 1:
            x, y, w, h = box
            img[y:y+h, x:x+w] = 255



    #print(pytesseract.image_to_string(img))
    #print(pytesseract.image_to_string(img, lang= 'rus'))



    dict_words, box = get_word_boxes(img, type_platform)
    #print(dict_words["text"])

    #print(box)

    if box is None:
        return "Загрузите другую фотографию"

    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w, y + h
    box_xyxy = [x1, y1, x2, y2]
    #print(box_xyxy)
    dict_words["box"] = xywh2xyxy(np.array(dict_words["box"]))




    out = find_digits(dict_words, box_xyxy)
    #print(len(out))



    if not(out is None): # Если нашли число

        return out



    else: # Если не нашли
        print("Поиск дальше...")
        # Детектим все заново
        img = cv2.imread(path, 0)
        if type_platform == "zn":
            img = image_preprocess(img, "")
        else:
            img = image_preprocess(img, type_platform)

        if type_platform == "vk":
            out = find_in_line(img, type_platform)
            if not(out is None): # Если нашли число
                return out

        #x, y, w, h = box
        if type_platform == "zn":
            k1, k2 = 0,  20
        elif type_platform == "tg":
            k1, k2 = 10,  15
        else:
            k1 = k2 = 10
        x1, y1, x2, y2 = 0, max(0, y - k1 * h), img.shape[1] - 1, min(y + k2*h, img.shape[0] - 1)

        roi = img[y1:y2, x1:x2]


        #print(pytesseract.image_to_string(roi, lang= 'rus'))
        #print(pytesseract.image_to_string(roi))

        box = x, y - y1, w, h
        dict_words, _ = get_word_boxes(roi, type_platform)
        #print(dict_words)

        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
        box_xyxy = [x1, y1, x2, y2]
        #print(box_xyxy)
        dict_words["box"] = xywh2xyxy(np.array(dict_words["box"]))





        ###print(dict_words)
        out = find_digits(dict_words, box_xyxy, thresh = 1.0)



    return out
