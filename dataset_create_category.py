#!/usr/bin/python

# 카테고리 = 의복종류

### IMPORTS
from __future__ import print_function

from config import *
from dataset_create_super import *


### GLOBALS
# 데이터셋의 의복 종류 개수 -----------
max_categories = 50

# 딥패션 데이터셋의 모든 의복 종류 -----------
# category_name_generate = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']

# 내가 분류할 의복 종류만 지정 ---------------
# category_name_generate = ['Tee', 'Top', 'Henley', 'Button-Down', 'Blouse', 'Flannel', 'Tank', 'Jersey', 'Hoodie', 'Chinos', 'Culottes', 'Jeans', 'Joggers', 'Sweatpants', 'Shorts', 'Sweatshorts', 'Cutoffs', 'Jeggings', 'Capris', 'Leggings', 'Skirt', 'Dress', 'Shirtdress', 'Sundress', 'Nightdress', 'Halter', 'Coat', 'Peacoat', 'Jacket', 'Anorak', 'Blazer', 'Bomber', 'Cardigan']
category_name_generate = ['Tee', 'Top', 'Blouse']


### FUNCTIONS

# Get category names list
# 의복 종류 목록 얻기 -------------------------------------------
def get_category_names():

    category_names = []

    # 현재 폴더에 있는 Anno의 list_category_cloth.txt에서 의복 종류 목록을 얻음
    with open(fashion_dataset_path + '/Anno/list_category_cloth.txt') as file_list_category_cloth:
        next(file_list_category_cloth)

        # strip함수는 문자열 양 끝에 있는 공백,개행문자를 제거함
        # replace함수로 공백문자를 '_'로 바꿈
        for line in file_list_category_cloth:
            word=line.strip()[:-1].strip().replace(' ', '_')
            category_names.append(word)

    return category_names


# Create category dir structure
# 각 의복 종류 폴더 생성 ----------------------------------------
def create_category_structure(category_names):
    for idx,category_name in enumerate(category_names):
        if category_name not in category_name_generate:
            logging.debug('Skipping category_names {}'.format(category_name))
            continue
        logging.debug('category_names {}'.format(category_name))

        # 각 의복 종류에 대한 폴더유무 확인, 없으면 폴더 생성
        # dataset_train_path, dataset_val_path, dataset_test_path은 config.py에 있는 변수임
        if idx < max_categories:
            # Train
            category_path_name=os.path.join(dataset_train_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)

            # Validation
            category_path_name=os.path.join(dataset_val_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)

            # Test
            category_path_name=os.path.join(dataset_test_path, category_name)
            logging.debug('category_path_name {}'.format(category_path_name))
            if not os.path.exists(os.path.join(category_path_name)):
                os.makedirs(category_path_name)


# 학습을 위해 라벨링 정보에 따라 각 라벨로 분리된 이미지 생성 --------------------------------------
def generate_dataset_images(category_names):

    count = 0

    # 라벨링 정보 얻기
    # with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_list_bbox_ptr:
    with open(fashion_dataset_path + '/Anno/list_bbox-copy.txt') as file_list_bbox_ptr:

        # 이미지파일 정보 얻기
        # with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_list_category_img:
        with open(fashion_dataset_path + '/Anno/list_category_img-copy.txt') as file_list_category_img:

            # 이미지파일명 비교를 위한 평가파일 가져오기
            # with open(fashion_dataset_path + '/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:
            with open(fashion_dataset_path + '/Eval/list_eval_partition-copy.txt', 'r') as file_list_eval_ptr:
                next(file_list_category_img) # 두번 쓰지 않으면 ValueError: invalid literal for int() with base 10: 'category_label' 발생
                next(file_list_category_img)

                idx_crop = 1

                # 이미지파일명 얻기
                for line in file_list_category_img:
                    line = line.split()
                    image_path_name = line[0]
                    logging.debug('image_path_name {}'.format(image_path_name))                                 # img/Tailored_Woven_Blazer/img_00000051.jpg
                    image_name = line[0].split('/')[-1]
                    logging.debug('image_name {}'.format(image_name))                                           # image_name img_00000051.jpg
                    image_full_name = line[0].replace('/', '_')
                    logging.debug('image_full_name {}'.format(image_full_name))                                 # img_Tailored_Woven_Blazer_img_00000051.jpg
                    image_category_index=int(line[1:][0]) - 1
                    logging.debug('image_category_index {}'.format(image_category_index))                       # 2

                    # 내가 지정한 의복종류가 아니면 continue
                    if category_names[image_category_index] not in category_name_generate:
                        logging.debug('Skipping {} {}'.format(category_names[image_category_index], image_path_name))
                        continue

                    # 학습을 위한 데이터셋 생성
                    if image_category_index < max_categories:
                        dataset_image_path = ''

                        # dataset_create_super.py에 있는 get_dataset_split_name 함수 호출
                        dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                        if dataset_split_name == "train":
                            dataset_image_path = os.path.join(dataset_train_path, category_names[image_category_index], image_full_name)
                        elif dataset_split_name == "val":
                            dataset_image_path = os.path.join(dataset_val_path, category_names[image_category_index], image_full_name)
                        elif dataset_split_name == "test":
                            dataset_image_path = os.path.join(dataset_test_path, category_names[image_category_index], image_full_name)
                        else:
                            logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                            exit(1)

                        logging.debug('image_category_index {}'.format(image_category_index))
                        logging.debug('category_names {}'.format(category_names[image_category_index]))
                        logging.debug('dataset_image_path {}'.format(dataset_image_path))

                        # dataset_create_super.py에 있는 get_gt_bbox 함수 호출
                        ## Get ground-truth bounding boxes
                        ## Origin is top left, x1 is distance from y axis;
                        ## x1,y1: top left coordinate of crop; x2,y2: bottom right coordinate of crop
                        gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)
                        logging.debug('Ground bbox:  gt_x1:{} gt_y1:{} gt_x2:{} gt_y2:{}'.format(gt_x1, gt_y1, gt_x2, gt_y2))

                        image_path_name_src = os.path.join(fashion_dataset_path, image_path_name)
                        logging.debug('image_path_name_src {}'.format(image_path_name_src))

                        # dataset_create_super.py에 있는 calculate_bbox_score_and_save_img 함수 호출
                        ## 학습을 위해 라벨링 정보에 따라 각 라벨로 분리된 이미지 생성
                        calculate_bbox_score_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)

                        # TODO: Also cropping in test set. Check if required
                        # shutil.copyfile(os.path.join(fashion_dataset_path, 'Img', image_path_name), dataset_image_path)

                        idx_crop = idx_crop + 1
                        logging.debug('idx_crop {}'.format(idx_crop))

                        # if idx_crop is 1000:
                        #     exit(0)

                    count = count+1
                    logging.info('count {} {}'.format(count, dataset_image_path))


if __name__ == '__main__':
    # dataset_create_super.py에 있는 create_dataset_split_structure 함수 호출
    # 데이터셋 폴더 생성
    # create_dataset_split_structure()

    category_names = get_category_names()
    # logging.debug('category_names {}'.format(category_names))

    # 지정된 각 의복 종류 폴더 생성
    # create_category_structure(category_names)
    generate_dataset_images(category_names)

    # dataset_create_super.py에 있는 display_clothing_data 함수 호출
    display_clothing_data()
