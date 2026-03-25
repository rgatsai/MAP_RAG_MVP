"""
數據獲取模組：負責從各種開放資料源抓取數據，或生成模擬數據。
- 天氣與空氣品質：Open-Meteo
- 地震資訊：USGS
- 地點資訊：OpenStreetMap (Overpass API)
"""

import httpx
import random
from datetime import datetime, timedelta
import asyncio
import math

import config


async def fetch_weather_and_aqi(city: str, lat: float = 25.033, lon: float = 121.565) -> tuple[dict|None, dict|None]:
    """同時並行執行天氣與空氣品質的抓取任務"""
    weather_task = fetch_weather(city, lat, lon)
    aqi_task = fetch_air_quality(city, lat, lon)
    
    results = await asyncio.gather(weather_task, aqi_task, return_exceptions=True)
    
    weather = results[0] if not isinstance(results[0], Exception) else None
    aqi = results[1] if not isinstance(results[1], Exception) else None
    
    return weather, aqi

async def fetch_weather(city: str, lat: float, lon: float) -> dict:
    """從 Open-Meteo 抓取目前的氣象資訊"""
    url = f"{config.OPEN_METEO_BASE_URL}/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,rain,wind_speed_10m,weather_code",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    current = data["current"]
    weather_code = current.get("weather_code", 0)

    return {
        "city": city,
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "rain": current.get("rain", 0) > 0,
        "rain_mm": current.get("rain", 0),
        "wind_speed": current["wind_speed_10m"],
        "weather_description": _weather_code_to_desc(weather_code),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

async def fetch_air_quality(city: str, lat: float, lon: float) -> dict:
    """抓取目前的空氣品質資訊 (AQI, PM2.5, PM10)"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm10,pm2_5",
        "timezone": "auto"
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
    current = data["current"]
    
    return {
        "city": city,
        "aqi_us": current.get("us_aqi"),
        "pm2_5": current.get("pm2_5"),
        "pm10": current.get("pm10"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

async def fetch_earthquakes(lat: float, lon: float, radius_km: int = 500) -> list[dict]:
    """從 USGS 抓取全球即時地震資訊"""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # 計算經緯度範圍
    lat_diff = radius_km / 111.0
    lon_diff = radius_km / 111.0
    
    params = {
        "format": "geojson",
        "minlatitude": lat - lat_diff,
        "maxlatitude": lat + lat_diff,
        "minlongitude": lon - lon_diff,
        "maxlongitude": lon + lon_diff,
        "starttime": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "minmagnitude": 3.0 # 只抓取規模 3.0 以上的地震
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    events = []
    for feature in data.get("features", [])[:5]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        
        time_str = datetime.fromtimestamp(props["time"]/1000.0).strftime("%Y-%m-%d %H:%M")
        
        events.append({
            "place": props["place"],
            "magnitude": props["mag"],
            "depth_km": coords[2],
            "time": time_str
        })
        
    return events


def _weather_code_to_desc(code: int) -> str:
    """將 WMO 天氣代碼轉換為中文描述"""
    mapping = {
        0: "晴天", 1: "大致晴朗", 2: "局部多雲", 3: "多雲",
        45: "霧", 48: "霧凇",
        51: "細雨", 53: "中雨", 55: "大雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "陣雨", 81: "中陣雨", 82: "大陣雨",
        95: "雷雨", 96: "雷雨伴冰雹", 99: "強雷雨伴冰雹",
    }
    return mapping.get(code, f"天氣代碼 {code}")


async def fetch_nearby_pois(lat: float, lon: float, city: str, radius_m: int = 1500) -> list[dict]:
    """利用 Overpass API 抓取鄰近的真實地點 (餐廳、咖啡廳、景點)"""
    url = "https://overpass-api.de/api/interpreter"
    
    # Overpass 查詢語法
    query = f"""
    [out:json][timeout:10];
    (
      node["amenity"~"cafe|restaurant|fast_food"](around:{radius_m},{lat},{lon});
      node["tourism"~"museum|gallery|viewpoint"](around:{radius_m},{lat},{lon});
    );
    out center 100;
    """
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, data=query)
        resp.raise_for_status()
        data = resp.json()
        
    results = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en")
        if not name:
            continue
            
        amenity = tags.get("amenity")
        tourism = tags.get("tourism")
        category = amenity if amenity else tourism
        if not category: category = "景點"
        
        # 計算距離 (公尺)
        dist = _haversine_distance(lat, lon, el.get("lat", lat), el.get("lon", lon))
        
        results.append({
            "name": name,
            "category": category,
            "lat": el.get("lat", lat),
            "lon": el.get("lon", lon),
            "distance_m": round(dist),
            "city": city,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        
    # 按距離排序並取前 30 名
    results.sort(key=lambda x: x["distance_m"])
    return results[:30]

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """計算兩個經緯度之間的距離"""
    R = 6371000 # 地球半徑 (公尺)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def generate_historical_data(city: str, days: int = 7) -> list[dict]:
    """生成歷史模擬數據 (用於測試)"""
    all_data = []
    base_date = datetime.now()

    for i in range(days):
        date = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        temp = random.uniform(12, 32)
        is_rain = random.choice([True, False, False])

        # 模擬天氣文檔
        weather_doc = {
            "city": city, "temperature": round(temp, 1), "humidity": round(random.uniform(50, 95), 1),
            "rain": is_rain, "rain_mm": round(random.uniform(0, 30), 1) if is_rain else 0,
            "wind_speed": round(random.uniform(2, 25), 1),
            "weather_description": "雨天" if is_rain else ("冷天" if temp < 20 else "晴天"),
            "date": date,
        }
        all_data.append({"type": "weather", "data": weather_doc})
        
        # 模擬空氣品質文檔
        aqi_doc = {
            "city": city, "aqi_us": int(random.uniform(20, 150)),
            "pm2_5": round(random.uniform(5, 55), 1), "pm10": round(random.uniform(10, 80), 1),
            "date": date
        }
        all_data.append({"type": "air_quality", "data": aqi_doc})

        # 模擬一個簡單的地點
        poi_doc = {
            "name": f"{city}歷史古蹟",
            "category": "museum",
            "lat": 25.0, "lon": 121.5,
            "distance_m": 120,
            "city": city,
            "date": date
        }
        all_data.append({"type": "poi", "data": poi_doc})

    return all_data
