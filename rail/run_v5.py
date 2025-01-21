import os
import cv2
from dotenv import load_dotenv
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn.utils import prune # 프루닝 적용
import torchvision.models as models
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from PIL import Image
from datetime import datetime

import time
import serial.tools.list_ports
import threading
import mysql.connector
import paramiko

from pyModbusTCP.server import ModbusServer
import socket

# MQTT Configuration
def get_wireless_ip():
    # 호스트 이름을 가져옵니다.
    hostname = socket.gethostname()

    # 호스트 이름을 IP 주소로 변환합니다.
    ip_address = socket.gethostbyname(hostname)

    return ip_address

wireless_ip = get_wireless_ip()
print(f"무선 네트워크의 IP 주소는 {wireless_ip} 입니다.")

server = ModbusServer(host=wireless_ip, port=502, no_block=True, ipv6=False)


# .env 파일에서 환경 변수 로드
load_dotenv(r"C:\Users\User\Desktop\rail\cv_v2\.env")

# 데이터셋 경로 설정
labels_file = r"C:\Users\User\Desktop\cv_v2\labels.txt"
model_path = r"C:\Users\User\Desktop\cv_v2\model\yolov5\best_model_0.005757.pt"  # 모델 경로
p_scr = r"C:\Users\User\Desktop\cv_v2\model\yolov5\pruned_scripted_model.pt"
capture_dir = r"C:\Users\User\Desktop\cv_v2\captures"


# GPU 또는 CPU 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 현재 사용 중인 디바이스: {device}")

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

# 컨베이어벨트 제어
def send_conveyor_speed(speed):
    if 0 <= speed <= 255:
        ser.write(f"CV_MOTOR={speed}\n".encode())
    else:
        print("0 ~ 255 사이의 값을 입력하세요")

# 로봇 팔 제어
def send_servo_1_angle(angle=80):
    if 60 <= angle <=130:
        ser.write(f"SERVO_1={angle} \n".encode())
    else:
        print("60~ 130 사이의 값을 입력하세요 ")

def send_servo_2_angle(angle=180):
    if 0 <= angle <=180:
        ser.write(f"SERVO_2={angle} \n".encode())
    else:
        print("0~ 180 사이의 값을 입력하세요")

def send_servo_3_angle(angle=100):
    if 30 <= angle <=120:
        ser.write(f"SERVO_3={angle} \n".encode())
    else:
        print("30~ 120 사이의 값을 입력하세요")

def send_catch_on_off(on_off):
    if on_off:
        ser.write("CATCH=ON\n".encode())
    else:
        ser.write("CATCH=OFF\n".encode())
# -------------------------------------------------------------------------------------------------------------------------------#



# MySQL 연결 설정
db_config = {
    "host": os.getenv("DB_HOST", "10.10.201.130"),
    "port": int(os.getenv("DB_PORT", 3306)),  # 기본값 3306
    "user": os.getenv("DB_USER", "user"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}
db_table = os.getenv("DB_TABLE")  # 테이블 이름

# MySQL 연결
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor(dictionary=True)

# 3. YOLOv5 다중 분류 모델 정의
class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

class YOLOv5MultiClass(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CustomBackbone()
        self.multi_class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.multi_class_head(x)
        return x


# 파일에서 라벨 번호와 이름 읽기
with open(labels_file, 'r') as file:
    classes = [line.strip().split(' ', 1)[1] for line in file]  # 라벨 이름만 추출

# 클래스 수 계산
num_classes = len(classes)    

# 저장된 모델 로드


if not os.path.exists(model_path):
    raise FileNotFoundError(f"[ERROR] 모델 파일을 찾을 수 없습니다: {model_path}")

# 모델 초기화 및 로드
model = YOLOv5MultiClass(num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"[INFO] 모델 로드 완료: {model_path}")

for module in model.modules():
    if isinstance(module, torch.nn.Linear):  # Linear Layer에 대해 프루닝
        prune.l1_unstructured(module, name='weight', amount=0.3)  # 30% 가중치 제거
        prune.remove(module, 'weight')  # 프루닝 후 가중치 재구성
print("[INFO] 모델 프루닝 적용")

# TorchScript 변환
tscr_model = torch.jit.script(model)

# 변환된 모델 저장

tscr_model.save(p_scr)
print(f"[INFO] TorchScript 모델 저장 : {p_scr}")

# 전처리 파이프라인 정의
transform = Compose([
    Resize((512, 512)),  # 이미지 크기 조정
    ToTensor(),          # 이미지를 Tensor로 변환
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
])
print("[INFO] 전처리 파이프라인 준비 완료")

# 캡쳐 이미지 저장 경로 설정 (전역 설정)

if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

#----------------------------------------------------------------------------------------------------------------------------------#

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
    select_query = f"""
    SELECT MAX(pdc) FROM {db_table} 
    WHERE prc LIKE %s
    """
    cursor.execute(select_query, (f"{current_date}-{base_prc}%",))
    result = cursor.fetchone()
    max_pdc = result["MAX(pdc)"] if result and "MAX(pdc)" in result else None

    # 기존 번호 확인 및 새 번호 시작 설정
    if max_pdc:
        last_number = int(max_pdc.split('-')[-1])
        production_count = last_number + 1
    else:
        production_count = 1

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
            try:
                # PDC 생성 및 MySQL 데이터 삽입
                p_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pdc = f"{prc}-{str(production_count).zfill(5)}"
                production_count += 1

                insert_query = f"""
                INSERT INTO {db_table} (prc, name, pdc, result, p_date, defect_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                data = (prc, name, pdc, result, p_date, defect_name)
                cursor.execute(insert_query, data)
                conn.commit()
                print(f"[INFO] 데이터 삽입: {data}")
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

        # SFTP 업로드 함수
        def upload_to_sftp(local_path, remote_path):
            try:
                hostname = os.getenv("SFTP_HOST", "")
                port = int(os.getenv("SFTP_PORT", 22))
                username = os.getenv("SFTP_USER", "")
                password = os.getenv("SFTP_PASSWORD", "")

                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(hostname, port, username, password)

                sftp = ssh.open_sftp()
                sftp.put(local_path, remote_path)
                print(f"[INFO] 파일 업로드 성공: {remote_path}")
                sftp.close()
                ssh.close()
                return True
            except Exception as e:
                print(f"[ERROR] 파일 업로드 실패: {e}")
                return False

        # 프레임 저장
        output_path = os.path.join(capture_dir, f"{pdc}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"[INFO] 디텍팅 프레임 저장: {output_path}")
        # SFTP 업로드
        remote_file_path = f"/{os.path.basename(output_path)}"  # 서버에서 저장될 경로
        upload_success = upload_to_sftp(output_path, remote_file_path)
        print(f"[INFO] 올리기 성공 : {upload_success}")

        # 화면에 프레임 출력
        cv2.imshow("Live Detection", frame)

        break
    return result
# -------------------------------------------------------------------------------------------------------------------------------

print("레일 준비 완료")
def init_robotarm():
    send_servo_1_angle(80)
    send_servo_2_angle(180)
    send_servo_3_angle(80)
    send_catch_on_off(False)

def send_lamp_red(on_off):
    if on_off:
        ser.write("LAMP_RED=ON\n".encode())
    else:
        ser.write("LAMP_RED=OFF\n".encode())

def send_lamp_yellow(on_off):
    if on_off:
        ser.write("LAMP_YELLOW=ON\n".encode())
    else:
        ser.write("LAMP_YELLOW=OFF\n".encode())

def send_lamp_green(on_off):
    if on_off:
        ser.write("LAMP_GREEN=ON\n".encode())
    else:
        ser.write("LAMP_GREEN=OFF\n".encode())


global count

count = 0

PS1_Status = 0
PS2_Status = 0
PS3_Status = 0
Converyor_Status = 0
Robot_Status = 0

Product_Count = 0
Good_Count = 0
Bad_Count = 0

def modbus_server():
    server.start()

def modbus_send_val():
    list = []
    list.append(Converyor_Status)
    server.data_bank.set_holding_registers(0, list)
    list = []
    list.append(PS1_Status)
    server.data_bank.set_holding_registers(1, list)
    list = []
    list.append(PS2_Status)
    server.data_bank.set_holding_registers(2, list)
    list = []
    list.append(PS3_Status)
    server.data_bank.set_holding_registers(3, list)
    list = []
    list.append(count)
    server.data_bank.set_holding_registers(4, list)
    list = []
    list.append(Good_Count)
    server.data_bank.set_holding_registers(5, list)
    list = []
    list.append(Bad_Count)
    server.data_bank.set_holding_registers(6, list)
    threading.Timer(0.5, modbus_send_val).start()


def init_lamp_check_result():
    
    #타워램프 초기화
    send_lamp_red(False)
    send_lamp_yellow(False)
    send_lamp_green(False)
    server.data_bank.set_holding_registers(9, [0])

    #검사상태 초기화
    server.data_bank.set_holding_registers(10, [0])
    
    # 결과 초기화
    server.data_bank.set_holding_registers(11, [0])   
    # 센서 초기화
    server.data_bank.set_holding_registers(12, [0])
    init_robotarm()

def move_ng_robotarm():
    print("물건 이동 시작")
    print("1. 물건 잡음")
    send_servo_1_angle(129)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(1.0)
    send_catch_on_off(True)
    time.sleep(2.0)

    print("2. 물건 올림")
    send_servo_1_angle(89)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(2.0)

    print("3. 물건 이동")
    send_servo_1_angle(89)
    send_servo_2_angle(50)
    send_servo_3_angle(80)
    time.sleep(2.0)

    print("4. 물건 내림")
    send_servo_1_angle(130)
    send_servo_2_angle(50)
    send_servo_3_angle(80)
    time.sleep(2.0)
    send_catch_on_off(False)
    time.sleep(2.0)

    print("5. 원위치")
    send_servo_1_angle(80)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(2.0)


def move_ok_robotarm():
    print("물건 이동 시작")
    print("1. 물건 잡음")
    send_servo_1_angle(129)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(1.0)
    send_catch_on_off(True)
    time.sleep(2.0)

    print("2. 물건 올림")
    send_servo_1_angle(89)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(2.0)

    print("3. 물건 이동")
    send_servo_1_angle(89)
    send_servo_2_angle(130)
    send_servo_3_angle(80)
    time.sleep(2.0)

    print("4. 물건 내림")
    send_servo_1_angle(130)
    send_servo_2_angle(130)
    send_servo_3_angle(80)
    time.sleep(2.0)
    send_catch_on_off(False)
    time.sleep(2.0)

    print("5. 원위치")
    send_servo_1_angle(80)
    send_servo_2_angle(175)
    send_servo_3_angle(80)
    time.sleep(2.0)

#최초실행
modbus_send_val()

# 쓰레드를 시작합니다.
t1 = threading.Thread(target=serial_read_thread)
t1.daemon = True
t1.start()

t2 = threading.Thread(target=modbus_server)
t2.daemon =True
t2.start()

init_robotarm()

time.sleep(2.0)
print("start")


# 전역 변수 초기화
serial_receive_date = ""
serial_receive_data = ""
processing = False
res = ""

try:
    while True:
        condi = server.data_bank.get_holding_registers(13, 1)[0]
        if(condi):
            print("시스템 작동시작")
            while condi:
                condi = server.data_bank.get_holding_registers(13, 1)[0]

                if not condi:
                    print("시스템 작동 중단 요청")
                    #시스템 초기화
                    server.data_bank.set_holding_registers(13, [0])
                    #컨테이너 벨트 초기화
                    server.data_bank.set_holding_registers(14, [0])
                    send_conveyor_speed(0)

                    #초기화
                    init_lamp_check_result()
                    break


                #투입쪽에 물건이 들어오면 컨베이어 동작
                if "PS_3=ON" in serial_receive_date:
                    #센서 1
                    server.data_bank.set_holding_registers(12, [1])
                    # 검사상태 대기
                    server.data_bank.set_holding_registers(10, [1])

                    # 타워램프 초기화
                    send_lamp_red(False)
                    send_lamp_yellow(False)
                    send_lamp_green(False)
                    
                    
                    #컨테이너 벨트 작동
                    server.data_bank.set_holding_registers(14, [1])
                    send_conveyor_speed(255)
                    
                    serial_receive_date = ""  # 데이터 초기화
                    processing = True  # 디텍팅 작업 준비
                    print("1 진입")

                elif "PS_3=OFF" in serial_receive_data:
                    #컨테이너 벨트 작동
                    server.data_bank.set_holding_registers(14, [1])
                    send_conveyor_speed(255)



                # 중앙센서 검출
                elif "PS_2=ON" in serial_receive_date and processing:
                    print("[INFO] PS_2 감지: 디텍팅 시작")

                    #센서 2
                    server.data_bank.set_holding_registers(12, [2])
                    #검사상태 검사
                    server.data_bank.set_holding_registers(10, [2])

                    #타워램프
                    server.data_bank.set_holding_registers(9, [2])

                    serial_receive_date = ""  # 데이터 초기화
                    time.sleep(0.55) 

                    #컨테이너 벨트 정지
                    server.data_bank.set_holding_registers(14, [2])
                    send_conveyor_speed(0)

                    send_lamp_red(False)
                    send_lamp_yellow(True)
                    send_lamp_green(False)
                    res = detect_live_video(model, transform, classes)  # 디텍팅 작업 실행

                    # 디텍팅 후 PS_2=OFF와 동일한 처리를 보장
                    print("[INFO] 디텍팅 완료: 컨베이어 재개")
                    #컨테이너 벨트 작동
                    server.data_bank.set_holding_registers(14, [1])
                    send_conveyor_speed(255)  # 컨베이어 재개
                    processing = False  # 디텍팅 작업 종료
                

                            

                elif "PS_2=OFF" in serial_receive_data:
                    #검사상태 완료
                    server.data_bank.set_holding_registers(10, [3])


                    #타워램프 초기화
                    send_lamp_red(False)
                    send_lamp_yellow(False)
                    send_lamp_green(False)
                    server.data_bank.set_holding_registers(9, [0])


                elif "PS_1=ON" in serial_receive_date:
                    #센서 3
                    server.data_bank.set_holding_registers(12, [3])

                    #검사상태 완료
                    server.data_bank.set_holding_registers(10, [3])

                    print(serial_receive_date)
                    serial_read_thread = ""

                    
                    if res == 'NG' :
                        #타워램프
                        send_lamp_red(True)
                        send_lamp_yellow(False)
                        send_lamp_green(False)
                        server.data_bank.set_holding_registers(9, [1])

                        # 결과 불량
                        server.data_bank.set_holding_registers(11, [1])

                        time.sleep(0.05)
                        #컨테이너 벨트 정지
                        server.data_bank.set_holding_registers(14, [2])
                        send_conveyor_speed(0)

                        #로봇팔 동작
                        move_ng_robotarm()

                        #초기화
                        init_lamp_check_result()


                    elif res == 'OK':
                        #검사상태 완료
                        server.data_bank.set_holding_registers(10, [3])

                        #타워램프
                        send_lamp_red(False)
                        send_lamp_yellow(False)
                        send_lamp_green(True)
                        server.data_bank.set_holding_registers(9, [3])

                        # 결과 양품
                        server.data_bank.set_holding_registers(11, [2])
                        
                        time.sleep(0.05)
                        #컨테이너 벨트 정지
                        server.data_bank.set_holding_registers(14, [2])
                        send_conveyor_speed(0)

                        #로봇팔 동작
                        move_ok_robotarm()

                        #초기화
                        init_lamp_check_result()

        else :
            time.sleep(0.1)
      

except KeyboardInterrupt:
    print("시스템 종료")

    #시스템 초기화
    server.data_bank.set_holding_registers(13, [0])
    #컨테이너 벨트 초기화
    server.data_bank.set_holding_registers(14, [0])
    send_conveyor_speed(0)

    #초기화
    init_lamp_check_result()

    #타워램프 초기화    
    server.data_bank.set_holding_registers(9, [0])
    send_lamp_red(False)
    send_lamp_yellow(False)
    send_lamp_green(False)

    #검사상태 초기화    
    server.data_bank.set_holding_registers(10, [0]) 

    # 결과 초기화
    server.data_bank.set_holding_registers(11, [0])  


    server.stop()
    send_catch_on_off(False)
    #컨테이너 벨트 초기화
    server.data_bank.set_holding_registers(14, [0])
    send_conveyor_speed(0)

    ser.close
    cap.release()
    cursor.close()
    conn.close()
    cv2.destroyAllWindows()
