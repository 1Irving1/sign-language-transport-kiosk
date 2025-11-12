import axios from "axios";

const instance = axios.create({
  baseURL: "http://localhost:8080/api", // 백엔드 서버 주소
});

// 인식 요청 함수 
export const recognizeSignLanguage = async (
  recognitionTarget: string,
  signLanguageData: string
) => {
  const res = await instance.post(`/signlanguage/recognize/${recognitionTarget}`, {
    signLanguageData,
    recognitionTarget,
  });
  return res.data;
};

export default instance;