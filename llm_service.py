"""
LLM service: sends RAG context + user question to Ollama for analysis.
"""

import json
import ollama as ollama_lib

import config


async def analyze(context_docs: list[dict], question: str) -> dict:
    """
    Send retrieved context + question to Ollama for intelligence analysis.
    Returns parsed analysis dict, or raw text if JSON parsing fails.
    """
    # Build context string from RAG results
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"[資料 {i}] {doc['text']}")

    context_str = "\n".join(context_parts)

    user_prompt = f"""以下是從資料庫中檢索到的相關資料：

{context_str}

---

使用者問題：{question}

請根據以上資料進行分析，找出資料之間的關聯性和有價值的洞察。
請以 JSON 格式回覆，包含：
- "title": 情報標題
- "summary": 核心發現（1-2 句話）
- "insights": 洞察列表，每個包含 "finding"（發現）和 "evidence"（證據）
- "recommendations": 建議列表（字串陣列）
- "confidence": 信心程度（"high" / "medium" / "low"）

只回覆 JSON，不要加任何其他文字或 markdown 標記。"""

    # Call Ollama
    response = ollama_lib.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": config.ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response["message"]["content"]

    # Try to parse JSON from LLM response
    parsed = _extract_json(raw_text)

    if parsed:
        return {
            "title": parsed.get("title", "情報分析"),
            "summary": parsed.get("summary", ""),
            "insights": parsed.get("insights", []),
            "recommendations": parsed.get("recommendations", []),
            "confidence": parsed.get("confidence", "medium"),
            "raw_context_count": len(context_docs),
            "question": question,
        }
    else:
        # Fallback: return raw text as summary
        return {
            "title": "情報分析",
            "summary": raw_text[:500],
            "insights": [],
            "recommendations": [],
            "confidence": "low",
            "raw_context_count": len(context_docs),
            "question": question,
        }


def check_ollama_status() -> dict:
    """Check if Ollama is running and model is available."""
    try:
        models = ollama_lib.list()
        model_names = [m.model for m in models.models]
        has_model = any(config.OLLAMA_MODEL in name for name in model_names)
        return {
            "status": "ok",
            "models": model_names,
            "target_model": config.OLLAMA_MODEL,
            "model_available": has_model,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "target_model": config.OLLAMA_MODEL,
            "model_available": False,
        }


def _extract_json(text: str) -> dict | None:
    """Try to extract JSON from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block within markdown code fences
    import re
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find anything that looks like a JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
