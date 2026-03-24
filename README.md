# 情報地圖平台 (Intelligence Map) API 使用教學

這是一套基於本地端 Ollama (qwen2.5:7b) 與 ChromaDB 的 RAG 智能分析系統。它可以抓取全球的天氣、空氣品質、地震及模擬商業資料，並透過 AI 進行總結與分析。

## 1. 啟動伺服器

請打開 Windows 的 PowerShell，輸入以下指令來啟動 API 伺服器：

```powershell
cd c:\Users\user\.gemini\antigravity\playground\fiery-perihelion
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

> **提示**：啟動後，您可以打開瀏覽器前往 `http://localhost:8000/docs`。FastAPI 會自動生成一個可互動的 Swagger UI 介面，讓您可以直接在網頁上點擊、測試所有的 API！

## 2. API 使用流程

### 第一步：收集情報 (Ingest)
首先，我們需要讓系統去抓取特定地點的最新外部資料，存入本地的 ChromaDB 知識庫中。

**API 路徑**：`POST http://localhost:8000/intel/ingest`

**請求範例 (JSON)**：
你可以自由開啟或關閉特定資料來源，只要修改 `true`/`false` 即可。
```json
{
    "city": "Taipei",
    "lat": 25.033,
    "lon": 121.565,
    "fetch_weather": true,
    "fetch_air_quality": true,
    "fetch_earthquakes": true
}
```

### 第二步：分析情報 (Analyze)
一旦知識庫裡有資料了，你就可以用自然語言問 AI 任何問題。AI 會先從知識庫撈出相關的資料，再進行統整。

**API 路徑**：`POST http://localhost:8000/intel/analyze`

**請求範例 (JSON)**：
```json
{
    "question": "分析一下台北市最近的天氣、空氣品質以及對商業營業額的潛在影響？",
    "top_k": 5
}
```
*`top_k` 是指讓 AI 參考最相關的前 5 筆資料。*

### 第三步：查看知識庫 (Data)
如果你想知道系統現在到底存了哪些最原始的資料，可以呼叫這個端點。

**API 路徑**：`GET http://localhost:8000/intel/data`


## 3. 其他實用功能

- **灌入測試資料**：`POST http://localhost:8000/intel/seed` (自動產生過去 7 天的歷史模擬資料供測試)
- **清空知識庫**：`POST http://localhost:8000/intel/reset` (當資料太亂時，直接清空 ChromaDB)
- **系統健康檢查**：`GET http://localhost:8000/health` (查看 Ollama 模型是否正常連線)
