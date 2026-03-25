# Ollama 設定
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ChromaDB 設定
CHROMA_PERSIST_DIR = "./chroma_data"
CHROMA_COLLECTION_NAME = "intel_reports"

# Open-Meteo API
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"

# RAG 設定
RAG_TOP_K = 5

# LLM 提示詞模板
ANALYSIS_SYSTEM_PROMPT = """你是一位專業且嚴謹的情報分析師。
你的任務是根據提供的「原始資料」，直接回答使用者的問題，並產出有價值的情報摘要。

規則：
1. 使用繁體中文回覆
2. **非常重要：必須嚴格針對「使用者問題」回答！** 
   - 如果使用者問「鹹酥雞」，但提供的資料裡只有「滷味」與「手搖飲」，你必須明確告知「目前資料庫中沒有找到符合條件的鹹酥雞資料」。
   - **絕對不可以**拿不相干的資料（如滷味）充數假裝回答。
3. 如果有找到相關資料，請分析資料之間的關聯（例如天氣與營業額的關係），並提出具體的洞察和建議。
4. 以 JSON 格式回覆，包含以下欄位：
   - title: 情報標題
   - summary: 核心發現（1-2 句話）
   - insights: 具體洞察列表
   - recommendations: 建議列表
   - confidence: 信心程度 (high/medium/low)
"""
