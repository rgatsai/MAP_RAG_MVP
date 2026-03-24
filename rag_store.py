"""
RAG storage layer using ChromaDB.
Handles embedding, storage, and retrieval of intelligence documents.
"""

import json
import chromadb

import config


# ---------------------------------------------------------------------------
# ChromaDB client (singleton)
# ---------------------------------------------------------------------------

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (lazy init)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def add_documents(documents: list[dict]) -> int:
    collection = get_collection()

    ids = []
    texts = []
    metadatas = []

    for i, doc in enumerate(documents):
        doc_type = doc.get("type", "unknown")
        data = doc.get("data", doc)

        text = _data_to_text(doc_type, data)

        import time
        date_key = data.get('date') or data.get('time', 'unknown')
        doc_id = f"{doc_type}_{date_key}_{int(time.time()*1000)}_{i}"

        ids.append(doc_id)
        texts.append(text)
        metadatas.append({
            "type": doc_type,
            "date": date_key,
            "city": data.get("city", data.get("place", "")),
            "raw_json": json.dumps(data, ensure_ascii=False),
        })

    if ids:
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    return len(ids)


def query(question: str, n_results: int = 5) -> list[dict]:
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })

    return output


def get_all_documents(limit: int = 100) -> dict:
    collection = get_collection()
    count = collection.count()

    if count == 0:
        return {"total": 0, "documents": []}

    results = collection.get(
        limit=min(limit, count),
        include=["documents", "metadatas"],
    )

    docs = []
    for i in range(len(results["ids"])):
        docs.append({
            "id": results["ids"][i],
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })

    return {"total": count, "documents": docs}


def reset_collection():
    global _client, _collection
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(config.CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    _collection = None
    _client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_to_text(doc_type: str, data: dict) -> str:
    """Convert structured data into natural language for embedding."""
    if doc_type == "weather":
        rain_text = f"降雨量 {data.get('rain_mm', 0)} mm" if data.get("rain") else "無降雨"
        return (
            f"天氣情報：{data.get('city', '')} 於 {data.get('date', '')} 的天氣狀況。"
            f"氣溫 {data.get('temperature', '')}°C，"
            f"濕度 {data.get('humidity', '')}%，"
            f"{rain_text}，"
            f"風速 {data.get('wind_speed', '')} km/h，"
            f"天氣描述：{data.get('weather_description', '')}。"
        )
    elif doc_type == "air_quality":
        return (
            f"空氣品質情報：{data.get('city', '')} 於 {data.get('date', '')} 的空氣品質狀況。"
            f"美國 AQI 指數 {data.get('aqi_us', '')}，"
            f"PM2.5 濃度 {data.get('pm2_5', '')} μg/m³，"
            f"PM10 濃度 {data.get('pm10', '')} μg/m³。"
        )
    elif doc_type == "earthquake":
        return (
            f"地震情報：{data.get('place', '')} 於 {data.get('time', '')} 發生了規模 {data.get('magnitude', '')} 的地震。"
            f"地震深度為 {data.get('depth_km', '')} 公里。"
        )
    elif doc_type == "poi":
        return (
            f"周邊景點情報：位於 {data.get('city', '')} 的真實地點「{data.get('name', '')}」。"
            f"類別屬於：{data.get('category', '')}，"
            f"距離查詢中心座標約 {data.get('distance_m', 0)} 公尺。"
        )
    else:
        return json.dumps(data, ensure_ascii=False)
