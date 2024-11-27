import React, { useState, useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import { useSearchParams } from "react-router-dom";
import "../../css/chat.css";
import { getAllCharacterInfo } from "../../apis/UserAPICalls";
import { sendMessageToAI } from "../../apis/ChatAPICalls";
import Message from "./Message";
import voiceButton from "../chat/voice.png";

const ChatRoom = ({ userId, conversationId }) => {
  const [searchParams] = useSearchParams();
  const charNo = searchParams.get("character_id"); // URL에서 charNo 추출
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isDescriptionVisible, setDescriptionVisible] = useState(false);
  const [imageUrl, setImageUrl] = useState("");

  const dispatch = useDispatch();
  const allCharacter = useSelector((state) => state.user.characters);

  // 캐릭터 정보 로드
  useEffect(() => {
    dispatch(getAllCharacterInfo());
  }, [dispatch]);

  // 현재 캐릭터 정보 추출
  const character = allCharacter?.find(
    (character) => String(character.charNo) === charNo
  );

  const imageCharaUrl = character
    ? `http://localhost:8080/api/v1/character${character.profileImage}`
    : "";
  const charName = character ? character.charName : "알 수 없음";
  const description = character ? character.description : "";

  // 채팅 기록 로드
  useEffect(() => {
    const fetchChatHistory = async () => {
      // if (!conversationId) return;
      try {
        const response = await fetch(
          // `http://localhost:8000/chat_message/${conversationId}`
          `http://localhost:8000/chat_message/1`
        );
        const data = await response.json();
        setMessages(data.messages || []);
      } catch (error) {
        console.error("채팅 기록 로드 오류:", error);
      }
    };

    fetchChatHistory();
  }, [conversationId]);

  const sendMessage = async () => {
    if (!input.trim()) return;
  
    const userMessage = { role: "user", content: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
  
    try {
      const aiResponse = await sendMessageToAI(input, charNo, charName);
      const aiMessage = { role: "ai", content: aiResponse.answer };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
  
      // AI 응답에서 키워드 감지
      if (aiResponse.answer) {
        console.log("AI 응답:", aiResponse.answer); // AI의 응답 내용
        handleKeywordDetection(aiResponse.answer); // 키워드가 있으면 이미지를 설정
      }
    } catch (error) {
      console.error("메세지 전송 오류:", error);
    } finally {
      setInput(""); // 입력값 초기화
    }
  };
  

  const queryRouting = {
    "기뻐": "/imageMsg/스폰지밥_해피.jpg",
    "슬퍼": "/imageMsg/스폰지밥_새드.jpg",
    "일하는 중": "/imageMsg/스폰지밥_일.jpg",
  };

  const handleKeywordDetection = (message) => {
    // 로그 추가: 전달된 메시지 확인
    console.log("메시지:", message);
  
    let matchedImageUrl = ""; // 기본적으로 빈 문자열로 설정
  
    // 키워드와 매칭되는 이미지를 찾음
    Object.keys(queryRouting).forEach((keyword) => {
      if (message.includes(keyword)) {
        console.log(`키워드 발견: ${keyword} -> 이미지 URL: ${queryRouting[keyword]}`);
        matchedImageUrl = queryRouting[keyword]; // 해당 키워드의 이미지를 설정
      }
    });
  
    // 이미지 URL이 매칭되었을 경우에만 상태 업데이트
    if (matchedImageUrl) {
      console.log("최종 이미지 URL:", matchedImageUrl); // 최종 설정된 이미지 URL
      setImageUrl(matchedImageUrl);
    } else {
      console.log("매칭된 키워드 없음, 이미지 URL 초기화");
      setImageUrl(""); // 키워드가 없으면 imageUrl을 비워서 이미지 표시 안함
    }
  };
  
  




  // 캐릭터 설명 토글
  const toggleDescription = () => {
    setDescriptionVisible(!isDescriptionVisible);
  };

  return (
    <div className="chat-room-chatRoom">
      <div className="chat-header-chatRoom">
        {character && (
          <>
            <img className="charaImg-chatRoom" src={imageCharaUrl} alt="캐릭터 이미지" />
            <p>
              {charName}
              <button onClick={toggleDescription}>
                {isDescriptionVisible ? "▲" : "▼"}
              </button>
            </p>
          </>
        )}
      </div>
      {isDescriptionVisible && (
        <div className="chat-chara-description-chatRoom">
          <p>{description}</p>
        </div>
      )}
      <div className="chat-messages-chatRoom">
        {messages.map((msg, index) => (
          <Message
            key={index}
            role={msg.role}
            content={msg.content}
            imageUrl={msg.role === "ai" ? imageUrl : ""} // 이미지 URL을 메시지에 전달
          />
        ))}
      </div>
      <div className="chat-input-chatRoom">
        <input
          type="text"
          placeholder="캐릭터에게 메세지를 보내보세요!"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>보내기</button>
        <div className="voice-button-chatRoom">
          <div className="back-voiceButton-chatRoom">
            <img src={voiceButton} alt="Voice Button" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatRoom;
