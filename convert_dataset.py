"""
"""
import json
import cv2
import matplotlib.pyplot as plt

images_location = './datasets/highway-cctv-footage-images/input_images/'
labels_location = './datasets/highway-cctv-footage-images/ground_truth_boxes/labelbox.json'

new_dataset_location = './datasets/converted/positive'


index = 0
with open(labels_location) as json_file:
    annotations = json.load(json_file)
    for vehicle in annotations:
        img = cv2.imread(images_location+vehicle['External ID'])
        cars = vehicle['Label']['Car']
        for car in cars:
            car = car['geometry']
            x = []
            y = []
            for point in car:
                x.append(point['x'])
                y.append(point['y'])
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            car_img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(f'{new_dataset_location}/image-{index}.jpg', car_img)
            index += 1

            # plt.imshow(car_img)
            # plt.show()
