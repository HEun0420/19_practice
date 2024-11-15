from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datetime import datetime, timedelta, timezone
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

load_dotenv()

# Flask 애플리케이션 설정
app = Flask(__name__)

# 데이터베이스 URL 설정
DATABASE_URL = "sqlite:///chat_history.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

# SQLAlchemy 모델 정의
Base = declarative_base()

# SQLChatMessageHistory DB 테이블 모델
class ChatMessage(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    content = Column(String, nullable=False)
    role = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __init__(self, session_id, content, role):
        self.session_id = session_id
        self.content = content
        self.role = role

# DB 테이블 생성
Base.metadata.create_all(engine)

# 채팅 기록 관리 객체
class SQLChatMessageHistory:
    def __init__(self, session_id, connection):
        self.session_id = session_id
        self.connection = connection
        self.session = Session()

    def get_messages(self):
        return self.session.query(ChatMessage).filter_by(session_id=self.session_id).all()

    def add_message(self, message, role):
        new_message = ChatMessage(session_id=self.session_id, content=message, role=role)
        self.session.add(new_message)
        self.session.commit()

# 세션 ID 설정
session_id = "sql_history"
chat_history = SQLChatMessageHistory(session_id=session_id, connection=engine)

# 웹 페이지 로더 설정
loader1 = WebBaseLoader(web_path=["https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B0","https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B4/%EC%9E%91%EC%A4%91%20%ED%96%89%EC%A0%81",
    "https://vclock.kr/time/%EC%84%9C%EC%9A%B8/"] )
loader2 = PyMuPDFLoader("data/대사집.pdf")
docs = loader1.load() + loader2.load()  # 두 개의 문서 로딩 후 합침

# 임베딩 및 텍스트 분할 설정
embedding = OpenAIEmbeddings()

# 벡터스토어 설정
vectorstore = FAISS.from_documents(documents=docs, embedding=embedding)
retriever = vectorstore.as_retriever()

# 현재 시간에 따라 성격 정의 (낮, 밤, 정오)
KST = timezone(timedelta(hours=9))
current_time = datetime.now(KST)
hour = current_time.hour
personality = "정오" if hour == 12 else "낮" if 6 <= hour < 18 else "밤"

# 프롬프트 템플릿 설정
prompt = PromptTemplate.from_template(
    f"""
    # Role
    - 너는 캐릭터를 흉내내는 챗봇이야. 너는 캐릭터를 흉내내면서 나와 대화를 해.

    # Persona
    - 당신은 일본 애니메이션 '일곱개의 대죄'에 나오는 일곱개의 대죄 기사단의 단원인 에스카노르이다. 
    - 당신은 밤이거나 평소 모습일 때는 매우 소심하고 (특히 멀린과 엮인 일에서라면) 다소 자기 비하적일 정도로 스스로를 낮추는 성격이며 존댓말을 한다.
    - 당신은 낮이 되면 성격이 180도 반전되어 진지하고 오만한 성격이 되며 존댓말과 반말을 섞어 사용하고 강해진다.
    - 정오에는 반말을 사용한다.
    - 동료들을 굉장히 아끼고, 그들과 함께 할 수 있는 것을 영광으로 생각한다.
    - 밤의 너는 낮의 너를 두려워하고, 낮의 너는 밤의 너를 약골이라 생각한다.

    - {'낮이라 오만한 말투' if personality == '낮' else '밤이라 공손한 말투' if personality == '밤' else '정오라 반말을 사용'}을 사용합니다.

    # Policy
    - 낮일 때는 최대한 답변을 짧게 해줘.
    
    # Example

    #Task
    -  질문에 대해 에스카노르의 입장으로 답변하세요.

    # Question
    {{question}}

    # Context
    {{context}}

    # Answer:
    """
)

# LLM 모델 설정
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 뉴스 체인 설정
news_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 루트 페이지 (채팅 화면)
@app.route("/")
def index():
    return render_template("chat.html")

# 채팅 처리
@app.route("/chat", methods=["POST"])
def chat():
    # 사용자 메시지 받기
    user_message = request.json.get("message")

    # 채팅 기록 불러오기
    chat_history_messages = chat_history.get_messages()  
    context = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history_messages]) 

    # 질문과 컨텍스트 결합
    question = user_message
    input_data = f"Question: {question}\nContext:\n{context}"

    # 모델 응답 생성
    bot_response = news_chain.invoke(input_data)

    # 'Answer:' 제거
    bot_response = bot_response.replace("Answer:", "").strip()  # 'Answer:' 텍스트를 제거하고 앞뒤 공백을 없앰

    # 채팅 기록 저장
    chat_history.add_message(user_message, "user")  # 위치 인수로 전달
    chat_history.add_message(bot_response, "bot")   # 위치 인수로 전달

    # 응답 반환
    return jsonify({"user_message": user_message, "bot_response": bot_response})

# 애플리케이션 실행
if __name__ == "__main__":
    app.run(debug=True)
