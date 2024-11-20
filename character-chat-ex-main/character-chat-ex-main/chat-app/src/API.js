import axios from 'axios';

const API_URL = "http://localhost:8000/chat"; // FastAPI URL

export const sendMessageToAI = async (question) => {
  // TODO: 유저가 캐릭터를 선택할 수 있도록 character_id, character_name를 받아와서 처리 가능하도록 변경
  const payload = {
    user_id: 1,
    conversation_id: 1,
    question,
    character_id: 1,
    character_name: "스폰지밥"
  };

  const response = await axios.post(API_URL, payload);
  return response.data;
};
