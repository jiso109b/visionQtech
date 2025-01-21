import os
import cv2
from dotenv import load_dotenv
from datetime import datetime
import torch
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from PIL import Image
from datetime import datetime

import time
import serial.tools.list_ports
import threading

from model import load_and_prepare_model, create_transform
from sql import connect_to_db, get_next_pdc, insert_detection_result
from sftp import upload_to_sftp

# GPU/CPU 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 환경 설정
labels_file = r"C:\Users\User\Desktop\rail\cv_v2\cv_v2\labels.txt"
capture_dir = r"captures"
original_model_path  = r"C:\Users\User\Desktop\rail\cv_v2\cv_v2\model\yolov5\best_model_0.002221.pt"
db_table = os.getenv("DB_TABLE")  # DB 테이블 이름

# 라벨 읽기 및 클래스 수 계산
with open(labels_file, 'r') as file:
    classes = [line.strip().split(' ', 1)[1] for line in file]
num_classes = len(classes)

# 모델 및 전처리 초기화
_, pruned_model_path = load_and_prepare_model(original_model_path, num_classes, device)
print(f"[INFO] 사용 중인 모델 경로: {pruned_model_path}")

# 프루닝 후 모델 로드
model = torch.jit.load(pruned_model_path, map_location=device)
model.eval()
print("[INFO] TorchScript 모델 로드 완료")

# 전처리 초기화
transform = create_transform()

# MySQL 연결
conn, cursor = connect_to_db()

# -------------------------------------------------------------------------------------------------------------------------------#
# 실시간 디텍팅 실행
cap = cv2.VideoCapture(1)  # 1번 카메라 (웹캠) 열기
if not cap.isOpened():
    raise RuntimeError("[ERROR] 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")

# FPS 설정
cap.set(cv2.CAP_PROP_FPS, 15)
print("[INFO] 실시간 영상 디텍팅 시작...")

# Arduino Uno를 찾아서 시리얼 포트에 연결합니다.
def connect_to_arduino_uno():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino Uno" in port.description:
            try:
                ser = serial.Serial(port.device, baudrate=9600)
                return ser
            except serial.SerialException:
                pass
    return None

ser = connect_to_arduino_uno()

# 시리얼통신 수신 쓰레드 함수
def serial_read_thread():
    global serial_receive_date
    while True:
        read_data = ser.readline().decode()
        serial_receive_date = read_data

# 쓰레드 시작
t1 = threading.Thread(target=serial_read_thread)
t1.daemon = True
t1.start()
serial_receive_date = ""

# 컨베이어벨트 제어
def send_conveyor_speed(speed):
    if 0 <= speed <= 255:
        ser.write(f"CV_MOTOR={speed}\n".encode())
    else:
        print("0 ~ 255 사이의 값을 입력하세요")

# 로봇 팔 제어
# def send_servo_1_angle(angle=80):
#     if 60 <= angle <=130:
#         ser.write(f"SERVO_1={angle} \n".encode())
#     else:
#         print("60~ 130 사이의 값을 입력하세요 ")

# def send_servo_2_angle(angle=180):
#     if 0 <= angle <=180:
#         ser.write(f"SERVO_2={angle} \n".encode())
#     else:
#         print("0~ 180 사이의 값을 입력하세요")

# def send_servo_3_angle(angle=100):
#     if 30 <= angle <=120:
#         ser.write(f"SERVO_3={angle} \n".encode())
#     else:
#         print("30~ 120 사이의 값을 입력하세요")

# def send_catch_on_off(on_off):
#     if on_off:
#         ser.write("CATCH=ON\n".encode())
#     else:
#         ser.write("CATCH=OFF\n".encode())


#--------------------------------------- 실시간 디텍팅

# 예측 처리 함수
def get_predictions(outputs, threshold=0.5):
    """시그모이드 후 threshold 기준으로 다중 라벨 예측"""
    probabilities = torch.sigmoid(outputs)
    return (probabilities > threshold).int(), probabilities

def detect_live_video(model, transform, classes):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("[ERROR] 웹캠을 열 수 없습니다.")

    # 'good' 클래스 인덱스
    try:
        good_class_index = classes.index("good")
    except ValueError:
        raise ValueError("[ERROR] 'good' 클래스가 클래스 리스트에 없습니다.")

    # 날짜별 초기 설정
    current_date = datetime.now().strftime("%Y%m%d")  # 현재 날짜
    base_prc = "FA02-002"  # 고정된 값
    prc = f"{current_date}-{base_prc}"  # 최종 PRC 값
    name = "임펠러"  # 고정된 이름

    # PDC 번호 계산 (현재 날짜 기준으로 가장 높은 번호 찾기)
    production_count = get_next_pdc(cursor, db_table, current_date, base_prc)
    print(f"[INFO] 현재 PDC 시작 번호: {production_count:05d}")

    # 마지막 저장 시간을 추적
    last_saved_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임을 읽을 수 없습니다.")
            break

        # 프레임 크기 확인
        frame_height, frame_width = frame.shape[:2]

        # ROI 계산 (중앙 부분, 크기)
        roi_width = int(frame_width * 0.45)  # ROI 너비: 전체 가로의 45%
        roi_height = int(frame_height * 0.65)  # ROI 높이: 전체 세로의 65%
        roi_x = (frame_width - roi_width) // 2  # 중심부 X 좌표
        roi_y = (frame_height - roi_height) // 2  # 중심부 Y 좌표

        # ROI 적용: 이미지의 중앙 부분만 추출
        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)  # ROI 표시

        # OpenCV 이미지를 PIL 이미지로 변환 (ROI만)
        image = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        last_saved_time = None

        # 모델 예측
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = outputs.softmax(dim=1).squeeze().cpu().numpy()

        # 상위 클래스 결정
        top_index = probabilities.argmax()
        defect_name = classes[top_index]
        confidence = probabilities[top_index]

        # 결과 판단 (OK/NG)
        result = "OK" if top_index == good_class_index else "NG"
        text_color = (0, 255, 0) if result == "OK" else (0, 0, 255)

        # 바운딩 박스 및 라벨 표시 (ROI 내부에 바운딩 박스 생성)
        predictions, probabilities = get_predictions(outputs)  # get_predictions 함수로 다중 라벨 예측
        for i, (pred, prob) in enumerate(zip(predictions.squeeze().cpu().numpy(), probabilities.squeeze().cpu().numpy())):
            if pred:  # 활성화된 클래스만 표시
                label_name = f"{classes[i]} ({prob:.2f})"
                
                # ROI와 동일한 크기로 바운딩 박스 좌표 설정
                start_point = (roi_x, roi_y)  # ROI 좌상단 좌표
                end_point = (roi_x + roi_width, roi_y + roi_height)  # ROI 우하단 좌표
                color = (0, 255, 0) if classes[i] == "good" else (0, 0, 255)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, start_point, end_point, color, 2)

                # 라벨 그리기 (ROI 내부 상단에 표시)
                label_position = (start_point[0], start_point[1] - 10)
                cv2.putText(frame, label_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # 현재 시간 초 단위로 추출
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # 중복 방지: 같은 초 내에 저장하지 않음
        if last_saved_time == current_time:
            print("[INFO] 동일한 초 내에 감지된 데이터, 저장 생략.")
        else:
            # 데이터 삽입 로직에서 중복 제거
            try:
                # PDC 생성 및 MySQL 데이터 삽입
                p_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pdc = f"{prc}-{str(production_count).zfill(5)}"
                production_count += 1

                # 데이터 준비
                data = (prc, name, pdc, result, p_date, defect_name)

                # sql.py의 함수 호출
                insert_detection_result(cursor, conn, db_table, data)
            except Exception as e:
                print(f"[ERROR] 데이터 삽입 실패: {e}")

            # NG 검출 시 이미지 저장
            if result == "NG":
                capture_path = os.path.join(capture_dir, f"NG_{current_time}.jpg")
                cv2.imwrite(capture_path, frame)
                print(f"[INFO] NG 디텍팅 - 클래스: {defect_name}, 확률: {confidence:.2f}, 저장 위치: {capture_path}")
                ng_info = f"NG Detected: {defect_name} ({confidence:.2f})"

            # 마지막 저장 시간을 현재 시간으로 갱신
            last_saved_time = current_time


        # 디텍팅 정보 프레임에 표시
        label = f"Class: {defect_name} ({confidence:.2f})"
        result_label = f"{result}: {defect_name}"
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, result_label, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        # 프레임 저장
        output_path = os.path.join(capture_dir, f"{result}_{current_time}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"[INFO] 디텍팅 프레임 저장: {output_path}")

        # SFTP 업로드 호출
        remote_file_path = f"/{os.path.basename(output_path)}"  # 루트 디렉토리에 저장
        success = upload_to_sftp(output_path, remote_file_path)
        print(f"[INFO] 서버 업로드 : {success}")

        # 화면에 프레임 출력
        cv2.imshow("Live Detection", frame)

        break

# -------------------------------------------------------------------------------------------------------------------------------


# 리소스 해제
send_conveyor_speed(0)

ser.close
cap.release()
cursor.close()
conn.close()
cv2.destroyAllWindows()

# 주요 실행 함수
def main():
    global serial_receive_date, processing

    # 시리얼 통신 및 초기화
    ser = connect_to_arduino_uno()
    send_conveyor_speed(0)

    # send_servo_1_angle(80)
    # send_servo_2_angle(180)
    # send_servo_3_angle(100)
    # send_catch_on_off(False)

    while True:
        # 투입쪽 센서 감지
        if "PS_3=ON" in serial_receive_date:
            send_conveyor_speed(255)
            print("[INFO] PS_3 감지: 컨베이어 동작")
            serial_receive_date = ""  # 데이터 초기화
            processing = True  # 디텍션 준비 완료

        # 중앙 센서 감지
        elif "PS_2=ON" in serial_receive_date and processing:
            print("[INFO] PS_2 감지: 디텍션 시작")
            serial_receive_date = ""  # 데이터 초기화
            time.sleep(0.55) 
            send_conveyor_speed(0)
            detect_live_video(model, transform, classes)  # 디텍션 실행

            # 디텍션 후 컨베이어 재개
            print("[INFO] 디텍팅 완료: 컨베이어 재개")
            send_conveyor_speed(255)
            processing = False

        # 출구 센서 감지
        elif "PS_1=OFF" in serial_receive_date:
            print("[INFO] PS_1 감지: 컨베이어 멈춤")
            send_conveyor_speed(0)
            serial_receive_date = ""  # 데이터 초기화

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 리소스 해제
    cleanup()

# 리소스 해제 함수
def cleanup():
    send_conveyor_speed(0)
    ser.close()
    cap.release()
    cursor.close()
    conn.close()
    cv2.destroyAllWindows()
    print("[INFO] 모든 리소스 해제 완료")

# 실행 진입점
if __name__ == "__main__":
    main()