from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory


# FastAPI 애플리케이션 설정
app = FastAPI()
    
load_dotenv()

loader1 = WebBaseLoader(
    web_path=[
        "https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B4",
        "https://namu.wiki/w/%EC%97%90%EC%8A%A4%EC%B9%B4%EB%85%B8%EB%A5%B4/%EC%9E%91%EC%A4%91%20%ED%96%89%EC%A0%81",
        "https://vclock.kr/time/%EC%84%9C%EC%9A%B8/"
    ]
)
loader2 = PyMuPDFLoader("data/대사집.pdf")
docs = loader1.load() + loader2.load()

# 임베딩
embeddings = OpenAIEmbeddings()

# Semantic Chunking for RAG
semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in docs])

# DB 생성
vectorstore = FAISS.from_documents(documents=semantic_chunks, embedding=embeddings)

# 벡터스토어에 있는 정보를 검색하고 생성
retriever = vectorstore.as_retriever()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta, timezone

# 한국 시간대 설정 (UTC+09:00)
KST = timezone(timedelta(hours=9))

# TODO: memory를 사용하면 중복되는 대사 사용을 줄일 수 있는지 확인
day_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        # Role
        - You are a chatbot imitating a specific character.

        # Persona
        - 당신은 낮의 에스카노르이다. 매우 자신감 넘치고 오만한 성격을 가지고 있으며, 강력한 힘을 자부하며 진지한 태도를 보인다.
        - 낮의 에스카노르는 동료들을 아끼지만, 자신의 힘과 능력에 자부심이 넘쳐 상대를 압도하는 태도를 취한다.
        - 대화할 때 반말과 간결한 말투를 섞어 사용하며, 당당하고 강렬한 어조를 유지한다.
        - 멀린을 사랑한다.
        - 호승심이 있는 캐릭터는 아니다.
        - Also: {relevant_info}

        # Personality Traits
        - 동료를 소중히 여기는 발언을 한다.
        - 오만한 말을 자주한다.
        
        # Policy
        - 답변을 짧고 강하게 해줘.
    
        # Tone
        - 존댓말을 위주로 쓰지만, 마지막 문장은 반말을 쓴다.
        - 진중한 톤으로 이야기한다.
    
        # example
        - 상대가 명령을 하면, 5번 중 1번은 '제게 명령하려하다니 거만함 MAX군요.'가 출력되도록 한다.
    
        # Task
        - 질문에 대해 낮의 에스카노르 입장으로 답변하세요.
        
        # Speech Style
        - 에스카노르는 오만한 성격으로 말을하나, 상대방을 배려하는 진중한 말을 합니다.

        
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

night_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        # Role
        - You are a chatbot imitating a specific character.

        # Persona
        - 당신은 밤의 에스카노르이다. 소심하고 자신감이 적으며, 특히 멀린과 관련된 일에서는 겸손하게 행동한다.
        - 낮의 강한 자신감과는 반대로, 밤의 에스카노르는 약간 소심하고 예의 바르며 겸손한 태도를 보인다.
        - 존댓말을 사용하고, 스스로를 낮추는 표현을 자주 사용한다.
        - 낮의 자신을 두려워한다.
        - Also: {relevant_info}

        # Policy
        - 공손하고 정중하게 답변해줘.

        # Task
        - 질문에 대해 밤의 에스카노르 입장으로 답변하세요.
        
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ]
)

# 시간대에 따른 프롬프트 선택 함수
def select_prompt_based_on_time():
    current_time = datetime.now(KST)
    hour = current_time.hour
    
    # 낮 (6시 ~ 18시)
    if 6 <= hour < 18:
        return day_prompt
    else:
        return night_prompt
    


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def get_response_chain():
    prompt = select_prompt_based_on_time()
    chain = (   # solution
        {
            "question": lambda x: x["question"], 
            "chat_history": lambda x: x["chat_history"], 
            "relevant_info": lambda x: retriever.get_relevant_documents(x["question"]) 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///chat_history.db"
    )
    
config_field = [
    ConfigurableFieldSpec(
        id="user_id",       # 설정 값의 고유 식별자
        annotation=str,     # 설정 값의 데이터 타입
        name="USER ID",     # 설정 값의 이름
        description="Unique identifier for a user", # 설정 값에 대한 설명
        default="",         # 기본 값
        is_shared=True      # 여러 대화에서 공유되는 값인지 여부
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="CONVERSATION ID",
        description="Unique identifier for a conversation",
        default="",
        is_shared=True
    )
]

chain_with_history = RunnableWithMessageHistory(
    get_response_chain(),
    get_session_history=get_chat_history,   # 대화 기록을 가져오는 user defined 함수
    input_messages_key="question",          # 입력 메세지 키
    history_messages_key="chat_history",    # 대화 기록 메세지의 키
    history_factory_config=config_field     # 대화 기록 조회 시 참조할 파라미터
)

# user1, conversation1
config = {"configurable":{"user_id":"user1", "conversation_id":"conversation1"}}

search_query = "안녕?"
relevant_info_result = retriever.get_relevant_documents(search_query)

# 체인 호출
chain_with_history.invoke(
    {"question": search_query, "relevant_info": relevant_info_result}, 
    config
)


@app.get("/")
async def index(request: Request):

# 채팅 처리
@app.post("/chat")
async def chat(request: Request):



# search_query = "나는 솔의눈이라고 해~~"
# relevant_info_result = retriever.invoke(search_query)

# # 체인 호출
# chain_with_history.invoke(
#     {"question": search_query, "relevant_info": relevant_info_result}, 
#     config
#     )


# 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)