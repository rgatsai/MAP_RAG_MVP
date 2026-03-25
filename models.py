"""
數據模型定義：使用 Pydantic 定義請求與回應的 Schema。
"""

from pydantic import BaseModel
from typing import Optional


# --- 請求模型 (Requests) ---

class IngestRequest(BaseModel):
    """匯入資料的請求模型"""
    city: str = "Taipei"
    lat: Optional[float] = 25.033
    lon: Optional[float] = 121.565
    fetch_weather: bool = True
    fetch_air_quality: bool = True
    fetch_earthquakes: bool = True
    fetch_pois: bool = True


class AnalyzeRequest(BaseModel):
    """分析情報的請求模型"""
    question: str
    top_k: int = 5


# --- 數據模型 (Data) ---

class WeatherData(BaseModel):
    """天氣數據"""
    city: str
    temperature: float
    humidity: float
    rain: bool
    rain_mm: float
    wind_speed: float
    weather_description: str
    date: str

class AirQualityData(BaseModel):
    """空氣品質數據"""
    city: str
    aqi_us: int
    pm2_5: float
    pm10: float
    date: str

class EarthquakeData(BaseModel):
    """地震數據"""
    place: str
    magnitude: float
    depth_km: float
    time: str


class POIData(BaseModel):
    """興趣點數據 (POI)"""
    name: str
    category: str
    lat: float
    lon: float
    distance_m: float
    city: str
    date: str


# --- 回應模型 (Responses) ---

class IntelInsight(BaseModel):
    """分析產出的單一洞察"""
    finding: str
    evidence: str


class AnalysisResponse(BaseModel):
    """分析介面的回應模型"""
    title: str
    summary: str
    insights: list[IntelInsight]
    recommendations: list[str]
    confidence: str
    raw_context_count: int
    question: str


class IngestResponse(BaseModel):
    """匯入介面的回應模型"""
    status: str
    documents_added: int
    city: str
    message: str
