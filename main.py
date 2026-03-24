"""
Intelligence Map RAG API — FastAPI main application.

Endpoints:
  POST /intel/ingest   — Fetch & store weather, air quality, earthuqakes, business
  POST /intel/analyze  — Query RAG + LLM analysis
  GET  /intel/data     — View stored documents
  GET  /health         — System health check
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
    status = llm_service.check_ollama_status()
    if status["status"] == "ok":
        print(f"[OK] Ollama connected. Model: {status['target_model']}, Available: {status['model_available']}")
    else:
        print(f"[WARN] Ollama not reachable: {status.get('error')}. LLM analysis will fail.")
    print(f"[INFO] ChromaDB documents: {rag_store.get_all_documents(limit=1)['total']}")
    yield


app = FastAPI(
    title="Intelligence Map RAG API",
    description="情報地圖平台 — AI 情報分析 API",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
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
    Fetch latest weather, AQI, earthquakes based on flags, + generate business data,
    then store everything in ChromaDB for RAG.
    """
    documents = []

    # 1. Fetch Weather and/or AQI
    weather, aqi = None, None
    try:
        if req.fetch_weather or req.fetch_air_quality:
            # Reusing the parallel fetcher if we want both
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
        print(f"Error fetching Weather/AQI: {e}")

    # 2. Fetch Earthquakes
    earthquakes = []
    try:
        if req.fetch_earthquakes:
            earthquakes = await data_fetcher.fetch_earthquakes(req.lat, req.lon)
            for eq in earthquakes:
                documents.append({"type": "earthquake", "data": eq})
    except Exception as e:
             print(f"Error fetching Earthquakes: {e}")

    # 3. Fetch nearby real POIs
    pois = []
    try:
        if req.fetch_pois:
            pois = await data_fetcher.fetch_nearby_pois(req.lat, req.lon, req.city)
            for poi in pois:
                documents.append({"type": "poi", "data": poi})
    except Exception as e:
        print(f"Error fetching POIs: {e}")

    # 4. Store in ChromaDB
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
    context_docs = rag_store.query(req.question, n_results=req.top_k)

    if not context_docs:
        raise HTTPException(
            status_code=404,
            detail="ChromaDB is empty. Please call /intel/ingest first to load data.",
        )

    try:
        result = await llm_service.analyze(context_docs, req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama LLM error: {e}")

    return result


@app.get("/intel/data")
async def list_data(limit: int = 50):
    return rag_store.get_all_documents(limit=limit)


@app.post("/intel/seed")
async def seed_data(days: int = 7, city: str = "Taipei"):
    docs = data_fetcher.generate_historical_data(city, days=days)
    count = rag_store.add_documents(docs)
    return {
        "status": "ok",
        "documents_added": count,
        "message": f"已灌入 {days} 天的歷史模擬資料（共 {count} 筆）",
    }


@app.post("/intel/reset")
async def reset_data():
    rag_store.reset_collection()
    return {"status": "ok", "message": "ChromaDB collection has been reset."}
