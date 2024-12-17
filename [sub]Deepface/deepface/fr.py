import cv2
from deepface import DeepFace
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont

employee_db = {
    # "김땡땡(마스크) 부장" : [cv2.imread('./face/deepface/images/ddang_mask.jpg')],
    # "김땡땡 부장" : [cv2.imread('./face/deepface/images/ddang_nomask.jpg')],
    "김지혜(마스크) 책임" : [cv2.imread('./face/deepface/images/wise_nomask.jpg')],
    "김지혜 책임" : [cv2.imread('./face/deepface/images/wise_mask.jpg')],
    # "김태운(마스크) 주임" : [cv2.imread('./face/deepface/images/nge_mask.jpg')],
    # "김태운 주임" : [cv2.imread('./face/deepface/images/nge_nomask.jpg')],
}

# 유사도 임계값 설정
SIMILARITY_THRESHOLD = 0.4

def recognize_employee(frame):
    for name, img_paths in employee_db.items():
        for img_path in img_paths:
            temp_img = frame

            try:
                result = DeepFace.verify(
                    img1_path=img_path,
                    img2_path=temp_img,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="ssd",
                )

                # print(f"비교 결과: {result}")

                if result["verified"] and result["distance"] < SIMILARITY_THRESHOLD:
                    print(f"인식되었습니다: {name}")

                    return name
            except ValueError as e:
                print("얼굴을 감지하지 못했습니다.", e)

                continue

    return None

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

font = ImageFont.truetype("./face/deepface/malgunbd.ttf", 30)

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라에서 프레임을 읽어올 수 없습니다.")

        break

    # 실시간 얼굴 인식 확인
    employee_name = recognize_employee(frame)

    if employee_name:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(pil_img)

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        draw.text((20, 20), f"{employee_name} / 출근확인", font=font, fill=(255, 0, 0))

        draw.text((20, 60), f"{time}", font=font, fill=(255, 0, 0))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 화면에 현재 프레임 표시
    cv2.imshow("Face Recognition", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()