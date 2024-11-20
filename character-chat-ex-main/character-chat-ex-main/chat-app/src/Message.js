import React from 'react';
import './index.css';
import ciracino from '../src/11sd.PNG';

const Message = ({ role, content }) => {
  return (
    <div>
      {role === 'ai' && (
        <div className="chat-charInfo-chatRoom">
          <img src={ciracino} alt="캐릭터 이미지" />
          <p>치라치노 (캐릭터 이름)</p>
        </div>
      )}
      <div className={`message-chatRoom ${role}`}>
        <div className={`message-bubble-chatRoom ${role}`}>
          {content}
        </div>
      </div>
    </div>
  );
};

export default Message;
