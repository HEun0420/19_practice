from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# 요청 데이터 모델
class QueryRequest(BaseModel):
    message: str

# 키워드와 이미지 매핑
keyword_image_map = {
    "SB_happy": "../static/example1.jpg",
    "SB_sad": "../static/example2.jpg",
    "SB_day": "../static/hello.jpg",
}

@app.post("/api/query")
async def process_message(query: QueryRequest) -> Dict:
    message = query.message
    response_text = process_input(message)

    # 키워드에 따른 이미지 선택
    matched_image = next(
        (url for keyword, url in keyword_image_map.items() if keyword in response_text),
        "/static/default.jpg"
    )

    return {"message": response_text, "imageUrl": matched_image}

def process_input(message: str) -> str:
    # 메시지 처리 및 응답 생성
    if "SB_happy" in message:
        return "This is a response for example1."
    elif "SB_sad" in message:
        return "Here's something related to example2."
    elif "SB_day" in message:
        return "Hello, welcome to the chatbot!"
    else:
        return "No specific keyword found, here's a default response."
