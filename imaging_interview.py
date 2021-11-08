import glob
import os
import argparse


import cv2
import imutils

parser = argparse.ArgumentParser(description='Dataset optimisation')
parser.add_argument('--path', help='path to dataset')
parser.add_argument('--min_len_cont',default=0, help='minimum length of contours that is checked for very similar images')
parser.add_argument('--min_prob',default=0, help='minimum probability that is checked for very similar images')
parser.add_argument('--minor_cont',default=10, help='no of contours checked for very minor changes')
parser.add_argument('--minor_prob',default=0.02, help='minumum probability that is checked for very minor changes')
parser.add_argument('--person_cont',default=2, help='no of contours checked for the presence of human or similar')
parser.add_argument('--person_prob',default=1, help='minumum probability that is checked for presence of human or similar')
parser.add_argument('--infront_cont',default=3, help='no of contours checked if something comes infront of the camera')
parser.add_argument('--infront_prob',default=30, help='minumum probability that is checked if something comes infront of the camera')
parser.add_argument('--car_cont',default=10, help='no of contours checked if a car comes in or leaves')
parser.add_argument('--car_prob',default=10, help='minumum probability that is checked if a car comes in or leaves')
parser.add_argument('--climatic_cont',default=15, help='no of contours checked due to changes in climatic conditions')
parser.add_argument('--climatic_prob',default=25, help='minumum probability that is checked due to changes in climatic conditions')


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


def main():
    args = parser.parse_args()

    if args.path is None:
        print('Enter path to directory')
        exit()

    image_names = {}
    for index, name in enumerate(sorted(glob.glob(args.path + '/*'))):
        image_names[index] = {'image_name': name, 'compare_status': None}

    no_changes = 0
    climatic_changes = 0
    minor_sunlight_changes = 0
    car_changes = 0
    infront_camera = 0
    people_changes = 0

    for key, values in image_names.items():
        print(f'Looping over images {key}')
        next_frame_index = key + 1
        if next_frame_index in image_names.keys():
            try:
                next_frame_img = image_names[next_frame_index]['image_name']
            except:
                pass

        image_1 = cv2.imread(values['image_name'])
        height, width = image_1.shape[0], image_1.shape[1]
        area_image = height * width
        preprocess_1 = preprocess_image_change_detection(image_1, gaussian_blur_radius_list=[3, 5])

        image_2 = cv2.imread(next_frame_img)
        preprocess_2 = preprocess_image_change_detection(image_2, gaussian_blur_radius_list=[3, 5])

        score, res_cnts, thresh = compare_frames_change_detection(preprocess_1, preprocess_2, min_contour_area=450)

        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        len_cnts = len(cnts)
        len_res_cnts = len(res_cnts)
        probability_change = score / area_image

        if len_res_cnts == args.min_len_cont or probability_change == args.min_prob:
            # This conditions checks whether there are no observable changes between two frames
            values['compare_status'] = True
            no_changes += 1

        elif len_cnts >= args.minor_cont and probability_change >= args.minor_prob:
            # This condition checks whether there are some minor changes mostly some changes in daylight
            values['compare_status'] = True
            minor_sunlight_changes += 1

        elif len_res_cnts < args.person_cont and probability_change < args.person_prob:
            # This condition checks whether there are any changes due to presence of humans or anthhing similar
            values['compare_status'] = False
            people_changes += 1

        elif len_res_cnts <= args.infront_cont and probability_change <= args.infront_prob:
            # This condition checks for presence of something right infront of the camera
            values['compare_status'] = False
            infront_camera += 1

        elif len_res_cnts < args.car_cont and probability_change <= args.car_prob:
            # This condition checks if there are changes due to presence or absence of cars from the scene
            values['compare_status'] = False
            car_changes += 1

        elif len_res_cnts >= args.climatic_cont or probability_change >= args.climatic_prob:
            # This condtion checks for changes that occur because of major climatic changes
            values['compare_status'] = True
            climatic_changes += 1
        else:
            # This condition caters all other changes and doesnot change the status to True for not to miss out crucial information
            values['compare_status'] = False


        if values['compare_status'] == True:
            os.remove(values['image_name'])

    count = 0
    for keys, values in image_names.items():
        if values['compare_status'] == True:
            count += 1


    print(f'Discarded images because of no observable changes {no_changes}')
    print(f'Discarded images because of climatic changes {climatic_changes}')
    print(f'Discarded images because of minor sunlight changes {minor_sunlight_changes}')
    print(f'Changes dye to vehicle movements {car_changes}')
    print(f'Changes due to movemnts infront of the camera {infront_camera}')
    print(f'Changes due to movemnts of person {people_changes}')
    print(f'Deleted images = {count}')
    print(f'Percentage of deleted images = {count / len(image_names) * 100}')


if __name__ == '__main__':
    main()
