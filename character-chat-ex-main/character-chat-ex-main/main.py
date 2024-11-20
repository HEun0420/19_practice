from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from chat_logic import setup_chat_chain
from models import ChatRequest, ChatResponse
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow only your React frontend during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# 캐릭터와 채팅
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        # chain을 캐릭터에 따라 set
        chat_chain = setup_chat_chain(request.character_id)
        
        config = {
            "configurable": {
                "user_id": request.user_id,
                "conversation_id": request.conversation_id
            }
        }

        response = chat_chain.invoke({"question": request.question}, config)
        return ChatResponse(answer=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history/{user_id}/{conversation_id}")
async def get_history(user_id: int, conversation_id: int, character_id: int, character_name: str):
    try:
        history = SQLChatMessageHistory(
            table_name="chat_history",
            session_id=conversation_id,
            connection=os.getenv("ENV_CONNECTION")
        )

        return {"messages": [
            {"role": "user" if msg.type == "human" else character_name, "content": msg.content}
            for msg in history.messages
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
