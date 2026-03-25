"""
情報地圖平台 — FastAPI 主應用程式
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

import data_fetcher
import rag_store
import llm_service
from models import (
    IngestRequest, IngestResponse,
    AnalyzeRequest, AnalysisResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化檢查
    status = llm_service.check_ollama_status()
    if status["status"] == "ok":
        print(f"[OK] Ollama 已連線。模型: {status['target_model']}, 可用狀態: {status['model_available']}")
    else:
        print(f"[WARN] 無法連線至 Ollama: {status.get('error')}。LLM 分析功能將受限。")
    print(f"[INFO] ChromaDB 目前文檔數: {rag_store.get_all_documents(limit=1)['total']}")
    yield


app = FastAPI(
    title="Intelligence Map RAG API",
    description="情報地圖平台 — AI 情報分析 API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """系統健康檢查"""
    ollama_status = llm_service.check_ollama_status()
    doc_count = rag_store.get_all_documents(limit=1)["total"]
    return {
        "status": "ok",
        "ollama": ollama_status,
        "chroma_documents": doc_count,
    }


@app.post("/intel/ingest", response_model=IngestResponse)
async def ingest_data(req: IngestRequest):
    """
    抓取最新的天氣、空氣品質、地震及鄰近 POI 資料，並存入 ChromaDB。
    """
    documents = []

    # 1. 抓取天氣與空氣品質
    weather, aqi = None, None
    try:
        if req.fetch_weather or req.fetch_air_quality:
            if req.fetch_weather and req.fetch_air_quality:
                weather, aqi = await data_fetcher.fetch_weather_and_aqi(req.city, req.lat, req.lon)
            else:
                if req.fetch_weather:
                    weather = await data_fetcher.fetch_weather(req.city, req.lat, req.lon)
                if req.fetch_air_quality:
                    aqi = await data_fetcher.fetch_air_quality(req.city, req.lat, req.lon)
                    
            if weather:
                documents.append({"type": "weather", "data": weather})
            if aqi:
                documents.append({"type": "air_quality", "data": aqi})
    except Exception as e:
        print(f"抓取天氣/空氣品質時發生錯誤: {e}")

    # 2. 抓取地震資料
    earthquakes = []
    try:
        if req.fetch_earthquakes:
            earthquakes = await data_fetcher.fetch_earthquakes(req.lat, req.lon)
            for eq in earthquakes:
                documents.append({"type": "earthquake", "data": eq})
    except Exception as e:
             print(f"抓取地震資料時發生錯誤: {e}")

    # 3. 抓取真實鄰近地點 (POI)
    pois = []
    try:
        if req.fetch_pois:
            pois = await data_fetcher.fetch_nearby_pois(req.lat, req.lon, req.city)
            for poi in pois:
                documents.append({"type": "poi", "data": poi})
    except Exception as e:
        print(f"抓取 POI 資料時發生錯誤: {e}")

    # 4. 存入 ChromaDB
    count = rag_store.add_documents(documents)

    msg = f"已匯入: "
    if weather: msg += "1 筆天氣, "
    if aqi: msg += "1 筆空氣品質, "
    if earthquakes: msg += f"{len(earthquakes)} 筆地震, "
    if pois: msg += f"{len(pois)} 筆真實鄰近地點(POI)資料"

    return IngestResponse(
        status="ok",
        documents_added=count,
        city=req.city,
        message=msg + " 到 ChromaDB",
    )


@app.post("/intel/analyze")
async def analyze_intel(req: AnalyzeRequest):
    """根據問題進行 RAG + LLM 分析"""
    context_docs = rag_store.query(req.question, n_results=req.top_k)

    if not context_docs:
        raise HTTPException(
            status_code=404,
            detail="ChromaDB 目前是空的。請先呼叫 /intel/ingest 匯入資料。",
        )

    try:
        result = await llm_service.analyze(context_docs, req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama LLM 錯誤: {e}")

    return result


@app.get("/intel/data")
async def list_data(limit: int = 50):
    """查看目前儲存的所有文檔"""
    return rag_store.get_all_documents(limit=limit)


@app.post("/intel/seed")
async def seed_data(days: int = 7, city: str = "Taipei"):
    """灌入歷史模擬資料清單"""
    docs = data_fetcher.generate_historical_data(city, days=days)
    count = rag_store.add_documents(docs)
    return {
        "status": "ok",
        "documents_added": count,
        "message": f"已灌入 {days} 天的歷史模擬資料（共 {count} 筆）",
    }


@app.post("/intel/reset")
async def reset_data():
    """重置 ChromaDB 集合"""
    rag_store.reset_collection()
    return {"status": "ok", "message": "ChromaDB 集合已重置。"}
