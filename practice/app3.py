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
        
class SQLChatMessageHistory:
    def __init__(self, session_id, connection):
        self.session_id = session_id
        self.connection = connection
        self.session = Session()  # 세션 초기화

    def get_messages(self):
        return self.session.query(ChatMessage).filter_by(session_id=self.session_id).all()

    def add_message(self, message, role):
        new_message = ChatMessage(session_id=self.session_id, content=message, role=role)
        self.session.add(new_message)
        self.session.commit()
        self.session.close()  # 세션을 닫아준다.


# 세션 ID 설정
session_id = "sql_history"
chat_history = SQLChatMessageHistory(session_id=session_id, connection=engine)

# 웹 페이지 로더 설정
loader1 = WebBaseLoader(web_path=["https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B0"] )
loader2 = PyMuPDFLoader("data/대사집.pdf")

docs = loader1.load() + loader2.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

# 임베딩 및 텍스트 분할 설정
embedding = OpenAIEmbeddings()

# 벡터스토어 설정
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()


# 프롬프트 템플릿 설정
day_prompt = PromptTemplate.from_template(
    f"""
    # Task
    - 질문에 대해 낮의 에스카노르 입장으로 답변하세요.
    
    # Role
    - 너는 캐릭터를 흉내내는 챗봇이야. 너는 '일곱개의 대죄'의 에스카노르가 된 것처럼 나와 대화해.

    # Persona
    - 당신은 낮의 에스카노르이다. 매우 자신감 넘치고 오만한 성격을 가지고 있다.
    - 강력한 힘을 자부하며 진지한 태도를 보인다.
    - 낮의 에스카노르는 동료들을 아끼지만, 자신의 힘과 능력에 자부심이 넘쳐 상대를 압도하는 태도를 취한다.
    - 대화할 때 반말과 간결한 말투를 섞어 사용하며, 당당하고 강렬한 어조를 유지한다.
    - 낮이 되면 성격이 180도 반전되어 진지하고 오만한 성격이 되며 존댓말과 반말을 섞어 사용하고 강해진다.
    - 동료를 다치게 하거나 무시하거나 상처입히면 화를 내고 용서하거나 사과를 원하지 않는다.
    - 낮에는 상대와 싸우면 "전 기분이 매우 좋습니다. 왜냐하면 내가 당신보다 한 수 위라는 것을 증명할 수 있는 절호의 기회니까요"라며 상대를 얕잡아본다.
    - 상대가 에스카노르에게 명령하면 '제게 명령하려하다니 거만함 MAX군요'라며 불쾌해하는 것을 5번 중 한번씩 출력한다.
    - 정오에는 반말을 사용한다.
    - 동료들을 굉장히 아끼고, 그들과 함께 할 수 있는 것을 영광으로 생각한다.
    - 동료를 다치게 하거나 무시하거나 상처입히면 화를 내고 용서하거나 사과를 원하지 않는다.
    
    
    # Policy
    - 답변을 짧고 강하게 해줘.


    # Question
    {{question}}

    # Context
    {{context}}
    
    # History
    {{history}}

    # Answer:
    """
)


night_prompt = PromptTemplate.from_template(
    f"""
    # Role
    - 너는 캐릭터를 흉내내는 챗봇이야. 너는 '일곱개의 대죄'의 에스카노르가 된 것처럼 나와 대화해.

    # Persona
    - 당신은 밤의 에스카노르이다. 소심하고 자신감이 적으며, 특히 멀린과 관련된 일에서는 겸손하게 행동한다.
    - 낮의 강한 자신감과는 반대로, 밤의 에스카노르는 약간 소심하고 예의 바르며 겸손한 태도를 보인다.
    - 존댓말을 사용하고, 스스로를 낮추는 표현을 자주 사용한다.
    - 밤의 너는 낮의 너를 두려워한다.
    - 동료들을 굉장히 아끼고, 그들과 함께 할 수 있는 것을 영광으로 생각한다.
    - 동료를 다치게 하거나 무시하거나 상처입히면 화를 내고 용서하거나 사과를 원하지 않는다.
    
    # Policy
    - 공손하고 정중하게 답변해줘.

    # Task
    - 질문에 대해 밤의 에스카노르 입장으로 답변하세요.

    # Question
    {{question}}

    # Context
    {{context}}
    
    # History
    {{history}}

    # Answer:
    """
)


# 시간대로 나눔
def select_prompt_based_on_time():
    KST = timezone(timedelta(hours=9))
    current_time = datetime.now(KST)
    hour = current_time.hour
    if 6 <= hour < 18:
        return day_prompt
    else:
        return night_prompt

# LLM 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5  )


# 뉴스 체인 설정
def get_response_chain():
    prompt = select_prompt_based_on_time()
    news_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return news_chain

# 루트 페이지 (채팅 화면)
@app.route("/")
def index():
    return render_template("chat3.html")

@app.route("/chat3", methods=["POST"])
def chat():
    # 사용자 메시지 받기
    user_message = request.json.get("message")

    # 채팅 기록 불러오기
    chat_history_messages = chat_history.get_messages()
    history = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history_messages])

    # 질문과 컨텍스트 결합
    # input_data = f"Question: {user_message}\nHistory:\n{history}"
    input_data =  {"question":user_message,
            "history":history,
        }

    # 체인으로 페르소나 나누기
    news_chain = get_response_chain()

    # 모델 응답 생성
    bot_response = news_chain.invoke(input_data)

    # 'Answer:' 제거 후, 마지막에 나온 답변만 반영되도록 처리
    if "Answer:" in bot_response:
        bot_response = bot_response.split("Answer:")[-1].strip()


    # 채팅 기록 저장
    chat_history.add_message(user_message, "user")  # 사용자 메시지 기록
    chat_history.add_message(bot_response, "bot")   # 챗봇 응답 기록

    # 응답 반환
    return jsonify({"user_message": user_message, "bot_response": bot_response})



# 애플리케이션 실행
if __name__ == "__main__":
    app.run(debug=True)
