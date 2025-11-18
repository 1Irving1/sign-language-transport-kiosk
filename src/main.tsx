//import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

//StrictMode의 문제가 있는지 검사하기 위해 두 번 실행하는 것으로 오류 발생, 주석 처리함
createRoot(document.getElementById('root')!).render(
  //<StrictMode>
    <App />
  //</StrictMode>,
)

