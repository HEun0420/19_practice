import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Redis 클라이언트 설정
r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# FastAPI 앱 설정
app = FastAPI()

# 템플릿과 정적 파일 설정
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 기본 페이지
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat6.html", {"request": request})

# 입력 받을 데이터 모델 (Pydantic)
class ChatMessage(BaseModel):
    message: str

@app.post("/chat/{user_id}")
async def chat(user_id: str, chat_message: ChatMessage):
    """
    사용자가 메시지를 보낼 때, 챗봇이 응답을 생성하고 대화 맥락을 저장합니다.
    """
    # 대화 맥락을 Redis에서 가져옴
    context_key = f"chat_session:{user_id}"
    current_context = r.get(context_key)
    
    # 현재 대화 맥락이 없다면, 첫 대화이므로 새로 시작
    if current_context:
        new_context = current_context + "\n" + chat_message.message
    else:
        new_context = chat_message.message
    
    # 챗봇의 간단한 응답 생성 (예시로 "You said: {message}"와 같은 방식으로)
    bot_response = f"You said: {chat_message.message}"
    
    # 새로운 대화 맥락을 Redis에 저장
    r.set(context_key, new_context)
    
    return {"user_message": chat_message.message, "bot_response": bot_response, "updated_context": new_context}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
