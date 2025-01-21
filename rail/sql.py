import os
import mysql.connector
from dotenv import load_dotenv

# .env 파일 로드 및 DB 연결
def connect_to_db():
    load_dotenv(r"C:.env")
    db_config = {
        "host": os.getenv("DB_HOST", "10.10.201.130"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "user"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME")
    }
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    print("[INFO] 데이터베이스 연결 완료")
    return conn, cursor

# PDC 번호 생성
def get_next_pdc(cursor, db_table, current_date, base_prc):
    select_query = f"""
    SELECT MAX(pdc) FROM {db_table} 
    WHERE prc LIKE %s
    """
    cursor.execute(select_query, (f"{current_date}-{base_prc}%",))
    result = cursor.fetchone()
    max_pdc = result["MAX(pdc)"] if result and "MAX(pdc)" in result else None

    if max_pdc:
        last_number = int(max_pdc.split('-')[-1])
        production_count = last_number + 1
    else:
        production_count = 1

    return production_count

# 데이터 삽입
def insert_detection_result(cursor, conn, db_table, data):
    insert_query = f"""
    INSERT INTO {db_table} (prc, name, pdc, result, p_date, defect_name)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, data)
    conn.commit()
    print(f"[INFO] 데이터 삽입 완료: {data}")
