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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()
app = Flask(__name__)


DATABASE_URL = "sqlite:///chat_history.db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


session_id = "sql_history"  


class SQLChatMessageHistory:
    def __init__(self, session_id, connection):
        self.session_id = session_id
        self.connection = connection
        self.messages = [] 

    def get_messages(self):
        return self.messages

    def add_message(self, message, role):
        self.messages.append({"content": message, "role": role})


chat_history = SQLChatMessageHistory(session_id=session_id, connection=engine)


loader1 = WebBaseLoader(web_path=["https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B4"])
loader2 = PyMuPDFLoader("data/대사집.pdf")
docs = loader1.load() + loader2.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
# splits = text_splitter.split_documents(docs)
embedding = OpenAIEmbeddings()

semantic_chunker = SemanticChunker(embedding, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in docs])

vectorstore = FAISS.from_documents(documents=semantic_chunks, embedding=embedding)
retriever = vectorstore.as_retriever()

# 밤낮
KST = timezone(timedelta(hours=9))
current_time = datetime.now(KST)
hour = current_time.hour
personality = "정오" if hour == 12 else "낮" if 6 <= hour < 18 else "밤"

prompt = PromptTemplate.from_template(
    f"""
    # Role
    - 너는 캐릭터를 흉내내는 챗봇이야

    # Persona
    - 당신은 일본 애니메이션 '일곱개의 대죄'에 나오는 일곱개의 대죄 기사단의 단원인 에스카노르이다. 
    - 당신은 밤이거나 평소 모습일 때는 매우 소심하고 (특히 멀린과 엮인 일에서라면) 다소 자기 비하적일 정도로 스스로를 낮추는 성격이며 존댓말을 한다.
    - 당신은 낮이 되면 성격이 180도 반전되어 진지하고 오만한 성격이 되며 존댓말과 반말을 섞어 사용하고 강해진다.
    - 정오에는 반말을 사용한다.
    - 동료들을 굉장히 아끼고, 그들과 함께 할 수 있는 것을 영광으로 생각한다.
    - 밤의 너는 낮의 너를 두려워하고, 낮의 너는 밤의 너를 약골이라 생각한다.

    - 밤에는 동료 외의 사람들이 명령을 내리면 비도덕적인게 아닌 이상 해준다.
    - 밤에는 자신의 힘을 두려워하고 스스로를 비하했다.
    - 동료를 다치게 하거나 무시하거나 상처입히면 화를 내고 용서하거나 사과를 원하지 않는다.
    - 낮에는 상대와 싸우면 "전 기분이 매우 좋습니다. 왜냐하면 내가 당신보다 한 수 위라는 것을 증명할 수 있는 절호의 기회니까요"라며 상대를 얕잡아본다.
    - 밤일 때는 낮을 언급하지 말고, 낮일 때는 밤을 언급하지 않는다.
    - {'낮이라 오만한 말투' if personality == '낮' else '밤이라 공손한 말투' if personality == '밤' else '정오라 반말을 사용'}을 사용합니다.

    # Policy
    - 낮일 때는 최대한 답변을 짧게 해줘.
    
    # Example

    #Task
    -  질문에 대해 에스카르노의 입장으로 답변하세요.

    # Question
    {{question}}

    # Context
    {{context}}

    # Answer:
    """
)


llm = ChatOpenAI(model="gpt-4", temperature=0)
news_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template("chat2.html")

@app.route("/chat2", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    
    chat_history_messages = chat_history.get_messages()  # 
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history_messages]) 

    
    question = user_message
    input_data = f"Question: {question}\nContext:\n{context}"

    
    bot_response = news_chain.invoke(input_data)

    
    chat_history.add_message(user_message, "user")  
    chat_history.add_message(bot_response, "bot")   
    
    return jsonify({"user_message": user_message, "bot_response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
