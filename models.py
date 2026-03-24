"""
Pydantic data models for request/response schemas.
"""

from pydantic import BaseModel
from typing import Optional


# --- Request Models ---

class IngestRequest(BaseModel):
    """Request to ingest data for a specific location."""
    city: str = "Taipei"
    lat: Optional[float] = 25.033
    lon: Optional[float] = 121.565
    fetch_weather: bool = True
    fetch_air_quality: bool = True
    fetch_earthquakes: bool = True
    fetch_pois: bool = True


class AnalyzeRequest(BaseModel):
    """Request to analyze intelligence."""
    question: str
    top_k: int = 5


# --- Data Models ---

class WeatherData(BaseModel):
    """Weather data from API."""
    city: str
    temperature: float
    humidity: float
    rain: bool
    rain_mm: float
    wind_speed: float
    weather_description: str
    date: str

class AirQualityData(BaseModel):
    """Air Quality data from Open-Meteo."""
    city: str
    aqi_us: int
    pm2_5: float
    pm10: float
    date: str

class EarthquakeData(BaseModel):
    """Earthquake data from USGS."""
    place: str
    magnitude: float
    depth_km: float
    time: str


class POIData(BaseModel):
    """Real point of interest from OpenStreetMap."""
    name: str
    category: str
    lat: float
    lon: float
    distance_m: float
    city: str
    date: str


# --- Response Models ---

class IntelInsight(BaseModel):
    """A single insight from analysis."""
    finding: str
    evidence: str


class AnalysisResponse(BaseModel):
    """Response from the analysis endpoint."""
    title: str
    summary: str
    insights: list[IntelInsight]
    recommendations: list[str]
    confidence: str
    raw_context_count: int
    question: str


class IngestResponse(BaseModel):
    """Response from the ingest endpoint."""
    status: str
    documents_added: int
    city: str
    message: str
