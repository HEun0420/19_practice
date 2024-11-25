import React, { useState } from 'react';
import { sendMessageToAI } from './API';
import Message from './Message';
import ciracino from '../src/11sd.PNG';
import voiceButton from '../src/voice.png';

const ChatRoom = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const [isDescriptionVisible, setDescriptionVisible] = useState(false);


  const toggleDescription = () => {
    setDescriptionVisible(!isDescriptionVisible);
  };


  // 메세지 보내기
  const sendMessage = async () => {
    if (input.trim() === "") return;

    const userMessage = { role: "user", content: input };
    setMessages([...messages, userMessage]);

    // 백엔드에서 메세지 전송 처리
    try {
      const aiResponse = await sendMessageToAI(input);
      const aiMessage = { role: "ai", content: aiResponse.answer };
      setInput("");
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      console.error("채팅 전송 에러:", error);
    }

    setInput("");
  };

  return (
    <div className="chat-room-chatRoom">
      <div className="chat-header-chatRoom">
        <img className='charaImg-chatRoom' src={ciracino}></img>
        <p>
          치라치노(캐릭터 이름)
          <button onClick={toggleDescription}>
            {isDescriptionVisible ? '▲' : '▼'}
          </button>
        </p>
      </div>
      {isDescriptionVisible && (
        <div className="chat-chara-description-chatRoom">
          <p>치라치노의 몸은 특별한 기름으로 뒤엎여 있어 펀치 등 상대의 공격을 받아넘긴다.
            하얀 털은 몸에서 나오는 기름으로 코딩되어 적의 공격도 매끄럽게 피한다. 전신에서 배어 나오는 기름은 매우 부드럽다.
            거친 피부가 고민인 사람에게도 효과적이다. </p>
        </div>
      )}
      <div className="chat-messages-chatRoom">
        {messages.map((msg, index) => (
          <Message key={index} role={msg.role} content={msg.content} />
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
