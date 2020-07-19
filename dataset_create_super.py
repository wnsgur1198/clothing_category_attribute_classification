#!/usr/bin/python

### IMPORTS
from __future__ import print_function

from config import *
from selective_search import selective_search_bbox


### FUNCTIONS

# 데이터셋 생성을 위한 폴더들이 없으면 생성 ----------
# dataset/train, dataset/test, dataset/validation
def create_dataset_split_structure():

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(dataset_train_path):
        os.makedirs(dataset_train_path)

    if not os.path.exists(dataset_val_path):
        os.makedirs(dataset_val_path)

    if not os.path.exists(dataset_test_path):
        os.makedirs(dataset_test_path)


# 데이터셋 폴더 경로명 추출 -----------
def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()


# 각 이미지파일명에 대응하는 라벨링된 경계박스 좌표 얻기 ------------------
def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox


# TODO: test this function
# http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Added due to comments on page
    if interArea < 0:
        interArea = 0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# def display_bbox(image_path_name, boxA, boxB):
#     logging.debug('image_path_name {}'.format(image_path_name))

#     # load image
#     img = skimage.io.imread(image_path_name)
#     logging.debug('img {}'.format(type(img)))

#     # Draw rectangles on the original image
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
#     ax.imshow(img)

#     # The origin is at top-left corner
#     x, y, w, h = boxA[0], boxA[1], boxA[2]-boxA[0], boxA[3]-boxA[1]
#     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
#     ax.add_patch(rect)
#     logging.debug('GT: boxA {}'.format(boxA))
#     logging.debug('   x    y    w    h')
#     logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

#     x, y, w, h = boxB[0], boxB[1], boxB[2]-boxB[0], boxB[3]-boxB[1]
#     rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
#     ax.add_patch(rect)
#     logging.debug('boxB {}'.format(boxB))
#     logging.debug('   x    y    w    h')
#     logging.debug('{:4d} {:4d} {:4d} {:4d}'.format(x, y, w, h))

#     plt.show()


# ??? --------------------------
def calculate_bbox_score_and_save_img(image_path_name, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2):

    logging.debug('dataset_image_path {}'.format(dataset_image_path))
    logging.debug('image_path_name {}'.format(image_path_name))

    # selective_search.py에 있는 selective_search_bbox 함수 호출
    candidates = selective_search_bbox(image_path_name)
    logging.debug('candidates {}'.format(candidates))

    image_name = image_path_name.split('/')[-1].split('.')[0]
    logging.debug('image_name {}'.format(image_name))

    img_read = Image.open(image_path_name)
    logging.debug('{} {} {}'.format(img_read.format, img_read.size, img_read.mode))

    i=0
    for x, y, w, h in (candidates):
        #  left, upper, right, and lower pixel; The cropped section includes the left column and
        #  the upper row of pixels and goes up to (but doesn't include) the right column and bottom row of pixels
        logging.debug('Cropped image: i {}'.format(i))
        i=i+1

        boxA = (gt_x1, gt_y1, gt_x2, gt_y2)
        boxB = (x, y, x+w, y+h)
        iou = bb_intersection_over_union(boxA, boxB)  # bb_intersection_over_union 함수 호출
        logging.debug('boxA {}'.format(boxA))
        logging.debug('boxB {}'.format(boxB))
        logging.debug('iou {}'.format(iou))

        # Uncomment only for testing as too much cpu/memory wastage
        #display_bbox(image_path_name, boxA, boxB)

        #img_crop = img_read.crop((y, x, y+w, x+h))
        img_crop = img_read.crop((x, y, x+w, y+h))

        image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
        image_save_path = dataset_image_path.rsplit('/', 1)[0]
        image_save_path_name = image_save_path + '/' + image_save_name + '_crop_' +  str(x) + '-' + str(y) + '-' + str(x+w) + '-' + str(y+h) + '_iou_' +  str(iou) + '.jpg'
        logging.debug('image_save_path_name {}'.format(image_save_path_name))
        # img_crop.save(image_save_path_name)
        # logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))

        img_crop_resize = img_crop.resize((w, h))
        img_crop_resize.save(image_save_path_name)
        logging.debug('img_crop_resize {} {} {}'.format(img_crop_resize.format, img_crop_resize.size, img_crop_resize.mode))

    # Ground Truth
    image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
    image_save_path = dataset_image_path.rsplit('/', 1)[0]
    image_save_path_name = image_save_path + '/' + image_save_name + '_gt_' +  str(gt_x1) + '-' + str(gt_y1) + '-' + str(gt_x2) + '-' + str(gt_y2) + '_iou_' +  '1.0' + '.jpg'
    logging.debug('image_save_path_name {}'.format(image_save_path_name))
    #img_crop = img_read.crop((gt_y1, gt_x1, gt_y2, gt_x2))
    img_crop = img_read.crop((gt_x1, gt_y1, gt_x2, gt_y2))
    img_crop.save(image_save_path_name)
    logging.debug('img_crop {} {} {}'.format(img_crop.format, img_crop.size, img_crop.mode))


# Display category and images count
def display_clothing_data():
    for path in [dataset_train_path, dataset_val_path, dataset_test_path]:
        logging.info('path {}'.format(path))

        # os.walk 함수 : 시작 디렉터리부터 시작하여 그 하위 모든 디렉터리를 차례대로 방문하게 해주는 함수
        # path1, dirs1, files1 = os.walk(path).next()                               # 파이썬3에서 지원하지 않는 형식임
        path1, dirs1, files1 = next(os.walk(path))
        
        file_count1 = len(files1)
        for dirs1_name in dirs1:
            # path2, dirs2, files2 = os.walk(os.path.join(path, dirs1_name)).next()  # 파이썬3에서 지원하지 않는 형식임
            path2, dirs2, files2 = next(os.walk(os.path.join(path, dirs1_name)))
            file_count2 = len(files2)
            logging.info('{:20s} : {}'.format(dirs1_name, file_count2))
