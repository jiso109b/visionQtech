from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pymysql
from openai import OpenAI

# .env 파일 로드
load_dotenv(r".env")

# Flask 애플리케이션 생성
app = Flask(__name__)

CORS(app)  # 모든 도메인에서 접근 허용

# OpenAI API 키 설정
client = OpenAI(api_key="API_KEY")

# DB 연결 함수
def get_connection():
    """
    데이터베이스 연결 함수
    """
    return pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )

def generate_query_with_gpt(question):
    """
    GPT를 사용하여 질문에서 SQL 쿼리를 생성
    """
    prompt = f"""
            아래 질문에 대해 **오직 SQL 코드만** 반환하세요. 불필요한 설명이나 주석은 포함하지 마세요.

            질문: "{question}"
            데이터베이스는 : "",
            데이터베이스 스키마: 
                1. ""
                2. ""
            
            테이블 간 관계:
                - ""

                **조건**:  
                    1. **SQL 반환 형식**:  
                    - **반환 형식은 순수 SQL 코드여야 하며, 불필요한 설명, 마크다운(예: ```sql)이랑 주석은 포함하지 마!**

                    2. **시간 범위 조건**:  
                    - 질문에 포함된 시간 범위(예: "지난 2일", "지난 1주일", "지난 한 달", "금일", "일주일")를 `WHERE` 절에 반영하세요.  
                        - "지난 1주일" => `INTERVAL 7 DAY`  
                        - "지난 2일" => `INTERVAL 2 DAY`  
                        - "한 달" => `INTERVAL 1 MONTH`  
                        - "금일" => `CURDATE()`  
                        - "전일" => `CURDATE() - INTERVAL 1 DAY`  

                    3. **불량률 계산 조건**:  
                    - `result = 'NG'`인 데이터의 합을 전체 데이터 개수로 나누어 불량률을 계산하세요.  
                    - SQL 쿼리에서 반드시 **`SUM(CASE WHEN ...)`과 `COUNT(*)`를 사용**하여 계산하세요.  
                        - 예: `SUM(CASE WHEN result = 'NG' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)`  
                    - `COUNT(CASE WHEN ...)` 대신 `SUM(CASE WHEN ...)`를 사용해야 합니다.

                    4. **제품명 조건**:  
                    - 제품명은 반드시 `name` 컬럼을 기준으로 필터링하세요.  
                    - 예: 특정 제품명이 `'임펠러'`인 데이터를 필터링하려면 `WHERE name = '임펠러'` 조건을 추가하세요. 
                    - 다른 제품도 마찬가지로 제품명('기어A', '기어B', '엔질벨트A', '엔진벨트B', '엔진벨브', '임펠러')을 넣어서 결과를 보여줘야 합니다. 

                    5. **예시 쿼리 조건**:  
                    -  불량률 제품별 계산 가능해야해.  
                    - 시간 조건은 `p_date >= CURDATE() - INTERVAL 1 DAY`를 사용.  

            예시 DB 구조:
                
            SQL 쿼리:
            """

    
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ],
        max_tokens=200,
        temperature=0.7
)
    # 메시지 내용 추출
    return completion.choices[0].message.content.strip()

def preprocess_sql_query(sql_query):
    """
    OpenAI에서 생성된 SQL 쿼리에서 불필요한 포맷 제거
    """
    return sql_query.replace("```sql", "").replace("```", "").strip()

# 데이터베이스 검색
def query_database(sql_query):
    """
    GPT에서 생성된 SQL 쿼리를 데이터베이스에서 실행
    """
    
    connection = get_connection()
    cursor = connection.cursor()
    
    try:
        # GPT에서 생성된 SQL 쿼리 실행
        cursor.execute(sql_query)
        results = cursor.fetchall()
    except Exception as e:
        # SQL 실행 중 오류 처리
        results = {"error": str(e)}
    finally:
        connection.close()

    return results

def generate_response_from_results(question, db_results):
    """
    데이터베이스 실행 결과를 GPT를 통해 자연어 응답으로 변환
    """
    if isinstance(db_results, dict) and "error" in db_results:
        # SQL 실행 중 오류가 발생한 경우
        db_text = f"SQL 실행 중 오류가 발생했습니다: {db_results['error']}"
    elif not db_results:
        # 결과가 없는 경우
        db_text = "관련된 데이터가 없습니다."
    else:
        # 결과를 텍스트로 변환
        db_text = "\n".join([str(row) for row in db_results])

    # GPT 프롬프트 생성
    prompt = f"""
    사용자의 질문: "{question}"

    데이터베이스에서 검색된 결과:
    {db_text}

    위 데이터를 바탕으로 사용자 질문에 대해 간결하고 명확하게 답변하세요.
    """
    
    # GPT 호출
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that uses database results to answer user questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return completion.choices[0].message.content.strip()

# Flask 엔드포인트
@app.route('/chat', methods=['POST'])
def chat():
    """
    사용자 요청을 처리하고 최종 응답 반환
    """
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # 1. GPT로 SQL 쿼리 생성
    sql_query = generate_query_with_gpt(question)

    # 2. 생성된 SQL 쿼리를 실행
    db_results = query_database(sql_query)

    # 3. SQL 결과를 기반으로 자연어 응답 생성
    response = generate_response_from_results(question, db_results)

    # 4. 최종 응답 반환
    return jsonify({
        'sql_query': sql_query,  # GPT가 생성한 SQL
        'db_results': db_results,  # SQL 실행 결과
        'response': response  # 최종 자연어 응답
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
