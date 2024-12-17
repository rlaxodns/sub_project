# https://github.com/KorniiVlasenko/Meters-Reader-YOLOv8-OpenCV

from ultralytics import YOLO

from utils import split_save_train_val_test, masks_to_labels, save_masked_by_yolo, cropp_and_save_masked
from utils import copy_matching_files, read_water_meter

if __name__ == '__main__':
    PREPARE_DATA_FOR_SEGMENT_TASK = False
    TRAIN_SEGMENTATION_MODEL = False

    PREPARE_DATA_FOR_DETECT_TASK = False
    TRAIN_DETECTION_MODEL = False

    MAKE_PREDICTION_ON_IMAGE = True

    if PREPARE_DATA_FOR_SEGMENT_TASK:
        path_to_dataset = './meters/data/original'
        path_to_save_train_val_test = './meters/data/segment'

        split_save_train_val_test(path_to_dataset, path_to_save_train_val_test)

        train_masks_path = './meters/data/segment/labels/train_masks'
        train_labels_path = './meters/data/segment/labels/train'

        masks_to_labels(train_masks_path, train_labels_path)

        val_masks_path = './meters/data/segment/labels/val_masks'
        val_labels_path = './meters/data/segment/labels/val'

        masks_to_labels(val_masks_path, val_labels_path)

        test_masks_path = './meters/data/segment/labels/test_masks'
        test_labels_path = './meters/data/segment/labels/test'

        masks_to_labels(test_masks_path, test_labels_path)

    if TRAIN_SEGMENTATION_MODEL:
        model = YOLO('yolo11n-seg')

        results = model.train(data='./meters/yolo_segment.yaml', epochs=10)

    if False:
        segment_model = YOLO('./runs/segment/train13/weights/best.pt')

        image_path = './meters/data/predict/id_1_value_13_116.jpg'

        results = segment_model.predict(image_path, show = True)

    if PREPARE_DATA_FOR_DETECT_TASK:
        path_to_images_dir = './meters/data/original/images'
        path_to_model = './runs/segment/train18/weights/best.pt'
        path_to_save_masked_images = './meters/data/masked'

        save_masked_by_yolo(path_to_images_dir, path_to_model, path_to_save_masked_images)

        path_to_masked_dir = './meters/data/masked'
        path_to_save_cropped = './meters/data/cropped'

        cropp_and_save_masked(path_to_masked_dir, path_to_save_cropped)

        path_to_cropped_images = './meters/data/cropped'
        path_to_labels = './meters/data/annotated'
        path_to_save_data = './meters/data/detection'

        copy_matching_files(path_to_cropped_images, path_to_labels, path_to_save_data)

    if TRAIN_DETECTION_MODEL:
        model = YOLO("yolo11n")

        results = model.train(data="./meters/yolo_detect.yaml", epochs=20)

    if MAKE_PREDICTION_ON_IMAGE:
        image_path = './meters/data/predict/test02.jpg' # Replace with a path to photo you want to read readings on
        segmentation_model_path = './meters/models/segment-best.pt'
        detection_model_path = './meters/models/detect-best.pt'
        path_to_save_predictions = './meters/predictions' # Replace with a path to folder where predictions will be saved 

        meters_readings = read_water_meter(image_path, segmentation_model_path, detection_model_path, path_to_save_predictions)