# https://github.com/KorniiVlasenko/Meters-Reader-YOLOv8-OpenCV

# 필요한 라이브러리 임포트
# YOLO 라이브러리 및 데이터 준비와 관련된 유틸리티 함수들을 불러옵니다.
from ultralytics import YOLO

from utils import split_save_train_val_test, masks_to_labels, save_masked_by_yolo, cropp_and_save_masked
from utils import copy_matching_files, read_water_meter

if __name__ == '__main__':
    # 데이터 준비, 모델 훈련, 예측 실행 여부 설정
    PREPARE_DATA_FOR_SEGMENT_TASK = False   # 세그멘테이션 작업을 위한 데이터 준비 여부
    TRAIN_SEGMENTATION_MODEL = False        # 세그멘테이션 모델 훈련 여부

    PREPARE_DATA_FOR_DETECT_TASK = False    # 탐지 모델을 위한 데이터 준비 여부
    TRAIN_DETECTION_MODEL = False           # 탐지 모델 훈련 여부

    MAKE_PREDICTION_ON_IMAGE = True         # 이미지에 대한 예측 수행 여부

    # 세그멘테이션 작업을 위한 데이터 준비
    if PREPARE_DATA_FOR_SEGMENT_TASK:
        # 원본 데이터셋 경로와 학습/검증/테스트 데이터셋을 저장할 경로 설정
        path_to_dataset = './Bon_project/sub_water_meter/meters/data/original'
        path_to_save_train_val_test = './Bon_project/sub_water_meter/meters/data/segment'

        # 원본 데이터셋을 학습, 검증, 테스트 세트로 나누어 저장합니다.
        split_save_train_val_test(path_to_dataset, path_to_save_train_val_test)

        # 학습 세트 마스크 -> 라벨 변환
        train_masks_path = './Bon_project/sub_water_meter/meters/data/segment/labels/train_masks'
        train_labels_path = './Bon_project/sub_water_meter/meters/data/segment/labels/train'
        masks_to_labels(train_masks_path, train_labels_path)

        # 검증 세트 마스크 -> 라벨 변환
        val_masks_path = './Bon_project/sub_water_meter/meters/data/segment/labels/val_masks'
        val_labels_path = './Bon_project/sub_water_meter/meters/data/segment/labels/val'
        masks_to_labels(val_masks_path, val_labels_path)

        # 테스트 세트 마스크 -> 라벨 변환
        test_masks_path = './Bon_project/sub_water_meter/meters/data/segment/labels/test_masks'
        test_labels_path = './Bon_project/sub_water_meter/meters/data/segment/labels/test'
        masks_to_labels(test_masks_path, test_labels_path)

    # 세그멘테이션 모델 훈련
    if TRAIN_SEGMENTATION_MODEL:
        # 세그멘테이션 모델 정의 및 훈련 실행
        model = YOLO('yolo11n-seg')

        results = model.train(data='./Bon_project/sub_water_meter/meters/yolo_segment.yaml', epochs=10)

# 세그멘테이션 모델을 사용해 예측 (이 코드는 실행되지 않음, 예시)
    if False:
        segment_model = YOLO('./runs/segment/train13/weights/best.pt')

        image_path = './meters/data/predict/id_1_value_13_116.jpg'

        results = segment_model.predict(image_path, show = True)

    # 탐지 작업을 위한 데이터 준비
    if PREPARE_DATA_FOR_DETECT_TASK:
        # 원본 이미지 디렉터리 및 세그멘테이션 모델 경로 설정
        path_to_images_dir = './Bon_project/sub_water_meter/meters/data/original/images'
        path_to_model = './runs/segment/train18/weights/best.pt'
        path_to_save_masked_images = './meters/data/masked'

        # 세그멘테이션 모델을 사용하여 이미지 마스킹 및 저장
        save_masked_by_yolo(path_to_images_dir, path_to_model, path_to_save_masked_images)

        # 마스킹된 이미지에서 숫자 영역을 잘라내고 저장
        path_to_masked_dir = './meters/data/masked'
        path_to_save_cropped = './meters/data/cropped'
        cropp_and_save_masked(path_to_masked_dir, path_to_save_cropped)

        # 잘라낸 이미지와 라벨을 탐지 데이터로 매칭하여 저장
        path_to_cropped_images = './Bon_project/sub_water_meter/meters/data/cropped'
        path_to_labels = './Bon_project/sub_water_meter/meters/data/annotated'
        path_to_save_data = './Bon_project/sub_water_meter/meters/data/detection'

        copy_matching_files(path_to_cropped_images, path_to_labels, path_to_save_data)
   
    # 탐지 모델 훈련
    if TRAIN_DETECTION_MODEL:
        # 탐지 모델 정의 및 훈련 실행
        model = YOLO("yolo11n")

        results = model.train(data="./Bon_project/sub_water_meter/meters/yolo_detect.yaml", epochs=20)

    # 이미지에 대한 예측 수행
    if MAKE_PREDICTION_ON_IMAGE:
        # 예측에 사용할 이미지 경로 및 모델 경로 설정
        image_paths = [
            f'./Bon_project/sub_water_meter/meters/data/predict/test{str(i).zfill(2)}.jpg'
            for i in range(1, 6)
        ]
        
        segmentation_model_path = 'C:/ai5/Bon_project/sub_water_meter/meters/models/segment-best.pt'
        detection_model_path = 'C:/ai5/Bon_project/sub_water_meter/meters/models/detect-best.pt'
        path_to_save_predictions = './Bon_project/sub_water_meter/meters/predictions' # Replace with a path to folder where predictions will be saved

        for image_path in image_paths:
            try:
                # 이미지에서 물 미터기 숫자 판독 수행
                meters_readings = read_water_meter(image_path, segmentation_model_path, detection_model_path, path_to_save_predictions)
                if meters_readings is not None:
                    print(f"Meter Readings for {image_path}: {meters_readings}")
            except AttributeError:
                print(f"Error: Could not process image: {image_path}. No detections found.")

        
        
        