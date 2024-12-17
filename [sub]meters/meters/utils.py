import cv2
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import shutil

# 3 FUNCTIONS FOR PREPARING DATA TO TRAINING SEGMENTATION MODEL

# 1. 데이터 준비: 이미지들을 학습/검증/테스트 데이터셋으로 나누어 저장하는 함수
# This function saves images to a specified directory
def save_images(images_paths, path_to_save): 
    # 디렉터리가 존재하지 않으면 생성
    # Create direcrory if doesn't exist
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # 각 이미지 저장
    # Save image
    for image_path in images_paths:
        image = cv2.imread(image_path)
        image_name = os.path.basename(image_path)
        file_path = os.path.join(path_to_save, image_name)

        # Check there is no image with same name yet
        if not os.path.exists(file_path):
            cv2.imwrite(file_path, image)

# 데이터셋을 학습/검증/테스트로 나누어 저장하는 함수
# 데이터 전처리 및 저장
def split_save_train_val_test(path_to_dataset, path_to_save_train_val_test): 
    # 원본 이미지와 마스크 경로 가져오기
    images_paths = glob.glob(os.path.join(path_to_dataset, 'images', "*"))
    masks_paths = glob.glob(os.path.join(path_to_dataset, 'masks', "*"))

    print(f'x len = {len(images_paths)}')
    print(f'y len = {len(masks_paths)}')

    # 데이터셋을 학습, 검증, 테스트로 나누기 (80%, 16%, 4% 비율)
    # Split dataset into train, validation, and test sets
    train_images_paths, val_images_paths, train_masks_paths, val_masks_paths = train_test_split(images_paths, masks_paths, test_size=0.2, random_state=0)
    val_images_paths, test_images_paths, val_masks_paths, test_masks_paths = train_test_split(val_images_paths, val_masks_paths, test_size=0.2, random_state=0)

    print(f'images_train len = {len(train_images_paths)}, images_val len = {len(val_images_paths)}, images_test len = {len(test_images_paths)}')
    print(f'masks_train len = {len(train_masks_paths)}, masks_val len = {len(val_masks_paths)}, masks_test len = {len(test_masks_paths)}')

    # 학습/검증/테스트 데이터를 지정된 폴더에 저장
    # Save each dataset
    save_images(train_images_paths, os.path.join(path_to_save_train_val_test, 'images', 'train'))
    save_images(train_masks_paths, os.path.join(path_to_save_train_val_test, 'labels', 'train_masks'))
    save_images(val_images_paths, os.path.join(path_to_save_train_val_test, 'images', 'val'))
    save_images(val_masks_paths, os.path.join(path_to_save_train_val_test, 'labels', 'val_masks'))
    save_images(test_images_paths, os.path.join(path_to_save_train_val_test, 'images', 'test'))
    save_images(test_masks_paths, os.path.join(path_to_save_train_val_test, 'labels', 'test_masks'))

# 2. 마스크 데이터를 YOLO용 라벨 형식으로 변환
# This function turns masks to label files that are neccessary for YOLO
def masks_to_labels(input_dir, output_dir): 
    # 출력 디렉터리가 없으면 생성
    # Create an output dir if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 각 마스크 파일을 라벨 파일로 변환
    for j in os.listdir(input_dir):
        image_path = os.path.join(input_dir, j)
        output_file_path = '{}.txt'.format(os.path.join(output_dir, j)[:-4])

        # 이미 파일이 존재하는 경우 건너뛰기
        if os.path.exists(output_file_path):
            continue

        # 이진 마스크를 불러와 컨투어 추출 후 다각형 변환
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어를 YOLO 형식의 라벨로 변환하여 저장
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # print the polygons
        with open(output_file_path, 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))

            f.close()

# 3. 세그멘테이션 모델을 사용하여 이미지에서 숫자 영역 마스킹
# 훈련된 YOLO 세그멘테이션 모델을 사용하여 이미지를 마스킹하고 저장합니다.
# 5 FUNCTIONS FOR PREPARING DATA TO TRAINING OBJECT DETECTION MODEL

# This function applies masking using trained segmentation model and saves masked images
def save_masked_by_yolo(path_to_images_dir, path_to_model, path_to_save_masked_images): 
    # YOLO 세그멘테이션 모델 로드
    # Load the trained segmentation model
    segment_model = YOLO(path_to_model)

    # 이미지 경로 및 이름 가져오기
    # Get images paths and names
    images_paths = glob.glob(os.path.join(path_to_images_dir, '*'))
    images_names = os.listdir(path_to_images_dir)

    # 출력 디렉터리가 없으면 생성
    # Create an output dir if doesn't exist
    if not os.path.exists(path_to_save_masked_images):
        os.makedirs(path_to_save_masked_images)

    # 모델로 마스킹을 적용하고 결과를 저장
    # Apply masking and save masked
    for i, image_path in enumerate(images_paths):

        image = cv2.imread(image_path)
        H, W, _ = image.shape

        results = segment_model(image)

        for result in results:
            for j, mask in enumerate(result.masks.data):
                # Transform predicted mask to the format suitable for opencv
                mask = mask.cpu().numpy() * 255
                mask = cv2.resize(mask, (W, H))
                mask = mask.astype(np.uint8)

                # apply masking
                masked = cv2.bitwise_and(image, image, mask = mask)

                # Get path where to save masked
                file_name = images_names[i]
                file_path = os.path.join(path_to_save_masked_images, file_name)

                # Write a file if it doesn't exist yet
                if not os.path.exists(file_path):
                    cv2.imwrite(file_path, masked)

# 4. 마스킹된 이미지에서 숫자 부분 자르기 및 회전
# This function croppes numbers area from one masked image
def crop_masked_image(masked_image): 
    # 이미지 그레이스케일 변환 및 블러링 후 컨투어 찾기
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    blured = cv2.GaussianBlur(gray, (7, 7), 0) # 노이즈를 덜 눈에 띄게 하기 위해 블러 처리
    _, binarized = cv2.threshold(blured, 1, 255, cv2.THRESH_BINARY) # 노이즈 제거
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 컨투어 찾기

    # 가장 큰 컨투어를 찾아 숫자 영역으로 자르기
    # Get the biggest contour (suppose it's rectangle around the numbers)
    largest_contour = max(contours, key=cv2.contourArea)

    gray_copy = gray.copy() # Copy for visualizations
    gray_copy = cv2.cvtColor(gray_copy, cv2.COLOR_GRAY2BGR)
    
    # Visualize biggest contour
    if False:
        cv2.drawContours(gray_copy, [largest_contour], -1, (255, 0, 0), 5)
        cv2.imshow('Largest Contour', gray_copy)

    # Find the smallest rectangle around our biggest contour
    rect = cv2.minAreaRect(largest_contour)

    # Get angles of the rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Visualize the bounding rectangle
    if False:
        cv2.drawContours(gray_copy, [box], -1, (0, 0, 255), 3)
        cv2.imshow('Rectangle', gray_copy)

    # Sort angles in order [top_left, top_right, bottom_left, bottom_right]
    box = box[box[:,1].argsort()]
    top_corners = box[:2][box[:2][:,0].argsort()]  # top_left, top_right
    bottom_corners = box[2:][box[2:][:,0].argsort()]  # bottom_left, bottom_right

    # Store angles in sorted order
    roi_sorted_corners = np.concatenate((top_corners, bottom_corners)) # returns order [top_left, top_right, bottom_left, bottom_right]

    # Visualize angles of rectangle
    if False:
        for i, corner in enumerate(roi_sorted_corners):
            cv2.circle(gray_copy, tuple(corner), 5, (0, 255, 0), -1)  # draw corners
            cv2.putText(gray_copy, f'{i}', tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow('ROI Sorted Corners', gray_copy)


    # Get shape of the image
    H, W = masked_image.shape[:2]

    # Angles of the image
    image_corners = [[0, 0], [W, 0], [0, H], [W, H]]


    # Deside in which side to rotate and rotate    
    if np.linalg.norm(np.array(roi_sorted_corners[0]) - np.array(roi_sorted_corners[1])) > np.linalg.norm(np.array(roi_sorted_corners[0]) - np.array(roi_sorted_corners[2])):

        # Original pixels of roi rectangle's angles
        pts1 = np.float32(roi_sorted_corners)

        # Desired (rotated) pixels of roi rectangle's angles
        pts2 = np.float32(image_corners)

        # Get matrix of transformation
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply transformation
        rotated_image = cv2.warpPerspective(gray, matrix, (2000, 2000)) 
        
        
    else:
        
        # Original pixels of roi rectangle's angles
        pts1 = np.float32(roi_sorted_corners)

        # Desired (rotated) pixels of roi rectangle's angles
        pts2 = np.float32([[0,W], [0, 0], [H, W], [H, 0]])

        # Get matrix of transformation
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply transformation
        rotated_image = cv2.warpPerspective(gray, matrix, (2000, 2000))  


    # Visualize rotated image
    if False:
        cv2.imshow('Rotated image', rotated_image)
    

    blured = cv2.GaussianBlur(rotated_image, (7, 7), 0) # Make noize less bright
    _, binarized = cv2.threshold(blured, 15, 255, cv2.THRESH_BINARY) # Get rid of noize 
    non_zero_pixels = cv2.findNonZero(binarized) # Find non-zero pixels on image without noize

    # Get bounding rectange for non-zero pixels
    x, y, w, h = cv2.boundingRect(non_zero_pixels)

    # Cut image using this bounding rectangle
    cropped_image = rotated_image[y:y+h, x:x+w]

    # Visualize cropped image
    if False:
        cv2.imshow('Cropped image', cropped_image)

    # Resize image in a way that will be convinient in further annotation
    resized_image = cv2.resize(cropped_image, (338, 83))

    # Visualize resized image
    if False:
        cv2.imshow('Resized image', resized_image)

    cv2.waitKey(0)

    return resized_image      

# 6. 마스킹된 이미지에서 모든 숫자 영역을 잘라내어 저장하는 함수
# This function croppes the area with numbers from all the masked images and saves cropped
def cropp_and_save_masked(path_to_masked_dir, path_to_save_cropped): 

    # Get paths and names of masked images
    images_paths = glob.glob(os.path.join(path_to_masked_dir, '*'))
    images_names = os.listdir(path_to_masked_dir)

    # 출력 디렉터리가 없으면 생성
    # Create an output dir if doesn't exist
    if not os.path.exists(path_to_save_cropped):
        os.makedirs(path_to_save_cropped)

    # 각 이미지에 대해 자르기 적용 후 결과 저장
    # Apply cropping and save cropped images
    for i, image_path in enumerate(images_paths):

        image = cv2.imread(image_path)
        cropped_image = crop_masked_image(image)

        file_name = images_names[i]
        file_path = os.path.join(path_to_save_cropped, file_name)

        if not os.path.exists(file_path):
            cv2.imwrite(file_path, cropped_image)

# 7. 파일들을 지정된 디렉터리에 복사하는 함수
# This function copies files to a specified directory
def save_files_to_dir(file_paths, path_to_save): 
    # 디렉터리가 없으면 생성
    # Create directory if doesn't exist
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    # 각 파일 복사하기
    for file_path in file_paths:

        file_name = os.path.basename(os.path.normpath(file_path))
        path_to_save_file = os.path.join(path_to_save, file_name)

        # Copy file, if doesn't exist yet
        if not os.path.exists(path_to_save_file):
            shutil.copy(file_path, path_to_save_file)

# 8. 객체 탐지 모델 학습을 위한 데이터 준비 (매칭 파일 복사)
# This function prepares data to training object detection model
def copy_matching_files(path_to_cropped_images, path_to_labels, path_to_save_data):  
    # 라벨 및 이미지 이름을 확장자 없이 가져오기
    # Get names of labels and images without extensions
    label_names = [os.path.splitext(label_name)[0] for label_name in os.listdir(path_to_labels)]
    image_names = [os.path.splitext(image_name)[0] for image_name in os.listdir(path_to_cropped_images)]

    
    # Variable for storing matching names
    matching_image_names = []

    for image_name in image_names:
        # If there is a label with same name, add to matching names
        if image_name in label_names:
            matching_image_names.append(image_name)
    
    # 라벨과 매칭되는 이미지 이름 저장
    # Bring extensions back
    matching_image_names = [image_name + '.jpg' for image_name in matching_image_names]
    label_names = [label_name + '.txt' for label_name in label_names]

    # Get matching image paths
    matching_image_paths = []
    for image_name in matching_image_names:
        image_path = os.path.join(path_to_cropped_images, image_name)
        matching_image_paths.append(image_path)

    # Get label paths
    label_paths = []
    for label_name in label_names:
        label_path = os.path.join(path_to_labels, label_name)
        label_paths.append(label_path)
    

    # Split images and labels path into train and val sets
    train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(matching_image_paths, 
                                                      label_paths, 
                                                      test_size = 0.2, 
                                                      random_state = 0)

    # Split val into val and test sets
    val_image_paths, test_image_paths, val_label_paths, test_label_paths = train_test_split(val_image_paths, 
                                                                        val_label_paths, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0)

    # YOLO에서 요구하는 폴더 구조에 맞춰 이미지 및 라벨 저장
    # Save imaages and labels in directory structure required by YOLO
    save_files_to_dir(train_image_paths, os.path.join(path_to_save_data, 'images', 'train'))
    save_files_to_dir(train_label_paths, os.path.join(path_to_save_data, 'labels', 'train'))
    save_files_to_dir(val_image_paths, os.path.join(path_to_save_data, 'images', 'val'))
    save_files_to_dir(val_label_paths, os.path.join(path_to_save_data, 'labels', 'val'))
    save_files_to_dir(test_image_paths, os.path.join(path_to_save_data, 'images', 'test'))
    save_files_to_dir(test_label_paths, os.path.join(path_to_save_data, 'labels', 'test'))
   
# 9. YOLO 모델의 객체 탐지 결과로부터 숫자를 읽어내는 함수
# 2 FUNCTIONS TO MAKE A PREDICTION

# This function converts object detection model's prediction to a number
def read_number(results_of_yolo_detection):

    # Get all numbers detected with bounding boxes
    numbers = []

    # Get this information about every digit [class, x_min, y_min, x_max, y_max]
    for result in results_of_yolo_detection:
        boxes = result.boxes  
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            predicted_class = box.cls[0].item()
            numbers.append((int(predicted_class), x_min.item(), y_min.item(), x_max.item(), y_max.item()))

    # Sort numbers in order from left to right
    sorted_numbers = sorted(numbers, key=lambda box: box[1])
    
    # Remove all zeros from the beginning
    while 0 < len(sorted_numbers):
        if sorted_numbers[0][0] == 0:
            sorted_numbers.remove(sorted_numbers[0])
        else:
            break

    # Remove all vertically duplicated detections 
    remove_list = []
    index = 0

    while index < len(sorted_numbers):

        # Search through all numbers but last two
        if index < (len(sorted_numbers) - 2):
            
            # Calculate distancies between current and next / next and after next numbers
            current_number_center = (sorted_numbers[index][1] + sorted_numbers[index][3]) / 2
            next_number_center = (sorted_numbers[index+1][1] + sorted_numbers[index+1][3]) / 2
            after_next_number_center = (sorted_numbers[index+2][1] + sorted_numbers[index+2][3]) / 2
            
            distance_to_next_number = next_number_center - current_number_center
            distance_to_after_next_number = after_next_number_center - next_number_center

            # 다음 숫자와의 거리가 그 다음 숫자와의 거리보다 훨씬 작다면 중복된 감지로 판단
            # If distance_to_next_number is significantly smaller than distance_to_after_next_number,
            # it means that one number is under another. We need to keep only one of them
            if distance_to_next_number < (distance_to_after_next_number / 2):

                # We will keep the number with larger height (The number that we see better)
                current_number_height = sorted_numbers[index][4] - sorted_numbers[index][2]
                next_number_height = sorted_numbers[index + 1][4] - sorted_numbers[index + 1][2]

                if current_number_height > next_number_height:
                    remove_list.append(index + 1)
                else:
                    remove_list.append(index)

        # Run though remaining numbers
        else:
            current_number_center = (sorted_numbers[index][1] + sorted_numbers[index][3]) / 2
            next_number_center = (sorted_numbers[index+1][1] + sorted_numbers[index+1][3]) / 2
            previous_number_center = (sorted_numbers[index-1][1] + sorted_numbers[index-1][3]) / 2

            distance_to_next_number = next_number_center - current_number_center
            distance_to_previous_number = current_number_center - previous_number_center

            if distance_to_next_number < distance_to_previous_number/2:
                
                current_number_height = sorted_numbers[index][4] - sorted_numbers[index][2]
                next_number_height = sorted_numbers[index + 1][4] - sorted_numbers[index + 1][2]

                if current_number_height > next_number_height:
                    remove_list.append(index + 1)
                    break
                else:
                    remove_list.append(index)
                    break

            break
        index += 1


    # Now we are going to remove all numbers from remove_list
    final_bounding_boxes = [box for idx, box in enumerate(sorted_numbers) if idx not in remove_list]

    # Transform detected bounding boxes to a number
    number = ''

    # Write every class (digit) to number string
    for box in final_bounding_boxes:
        number += str(box[0])

    # If len < 6, meter readings is an integer. Else, it has 3 digits after comma
    # See README.md for explanation
    if len(number) < 6: 
        number = int(number)
    else:
        number = int(number) / 1000

    return number

# 10. 하나의 이미지를 입력받아 예측 결과를 읽어내는 함수
# This function is made for making a prediction on one photo conviniently
def read_water_meter(image_path, segmentation_model_path, detection_model_path, path_to_save_predictions):

    # Cropp numbers zone from image
    seg_model = YOLO(segmentation_model_path) # Load trained segmentation model
    img = cv2.imread(image_path) 
    H, W, _ = img.shape 

    results = seg_model(img) # Cropp the numbers zone from water meter image

    # Get cropped numbers zone for numbers detection
    for result in results:
        for _, mask in enumerate(result.masks.data):

            # Convert mask to format I can use with opencv
            mask = mask.cpu().numpy() * 255
            mask = cv2.resize(mask, (W, H))      
            mask = mask.astype(np.uint8)
                
            # Apply masking
            masked = cv2.bitwise_and(img, img, mask = mask)

            # Cropp and rotate numbers zone for further numbers detection
            cropped = crop_masked_image(masked)
            cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Detection part
    det_model = YOLO(detection_model_path)
    results = det_model.predict(source=cropped, iou=0.7)

    # Read a number from detection model's prediction
    meter_readings = read_number(results)

    # 예측 결과를 이미지에 표시 및 저장
    # Create a folder for storing predictions, if doesn't exist
    if not os.path.exists(path_to_save_predictions):
        os.makedirs(path_to_save_predictions)

    # Write prediction right on the image
    cv2.putText(img, str(meter_readings), (20, H-50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 10)
    cv2.putText(img, str(meter_readings), (20, H-50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 7)

    # Show the result
    cv2.imshow('Predicted meters readigs', img)
    # cv2.imshow('Predicted meters readigs', cv2.resize(img, (img.shape[1]//3, img.shape[0]//3)))

    # Save the result
    file_name = os.path.basename(os.path.normpath(image_path))
    file_path = os.path.join(path_to_save_predictions, file_name)
    cv2.imwrite(file_path, img)
    
    # Print the result
    print(f'Meter Readings: {meter_readings}')

    cv2.waitKey(0)

    return meter_readings
