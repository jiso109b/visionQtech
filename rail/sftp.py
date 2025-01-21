import paramiko
import os
from dotenv import load_dotenv


def upload_to_sftp(hostname, port, username, password, local_file_path):

    try:

        # 파일 이름 추출하여 서버 저장 경로 생성
        remote_file_path = f"/{os.path.basename(local_file_path)}"

        # SSH 클라이언트 생성 및 연결
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # 호스트 키 자동 승인
        ssh.connect(hostname, port, username, password)            # 서버 연결

        # SFTP 세션 열기
        sftp = ssh.open_sftp()

        # 파일 업로드
        sftp.put(local_file_path, remote_file_path)
        print(f"파일 업로드 성공: {remote_file_path}")

        # SFTP 세션 종료
        sftp.close()
        ssh.close()
        return True
    except Exception as e:
        print(f"파일 업로드 실패: {e}")
        return False
    
if __name__ == "__main__":

    # .env 파일에서 SFTP 연결 정보 로드
    load_dotenv(r"C:\Users\User\Desktop\cv_v2\.env")  # .env 파일 경로 설정
    hostname = os.getenv("SFTP_HOST", "10.10.201.125")  # 서버 IP 주소 또는 도메인
    port = int(os.getenv("SFTP_PORT"))  # 기본 SFTP 포트: 22
    username = os.getenv("SFTP_USER", "mes")  # SFTP 사용자 이름
    password = os.getenv("SFTP_PASSWORD")  # SFTP 사용자 비밀번호

    # 로컬 파일 경로
    local_file_path = r"C:\Users\User\Desktop\cv_v2\captures"  # 업로드할 파일 경로

    # SFTP 업로드 호출
    success = upload_to_sftp(hostname, port, username, password, local_file_path)
    print(f"[INFO] 서버 업로드 : {success}")
