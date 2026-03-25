"""
種子資料腳本：預先將模擬的歷史數據導入 ChromaDB。
執行方式：python seed_data.py
"""

import data_fetcher
import rag_store


def main():
    print("🌱 正在將 7 天的模擬數據匯入 ChromaDB...")

    # 生成 7 天的歷史天氣與地點數據
    docs = data_fetcher.generate_historical_data("Taipei", days=7)
    print(f"   已生成 {len(docs)} 筆文檔")

    # 存入 ChromaDB
    count = rag_store.add_documents(docs)
    print(f"   已將 {count} 筆文檔存入 ChromaDB")

    # 驗證
    info = rag_store.get_all_documents(limit=1)
    print(f"   集合中的文檔總數: {info['total']}")
    print("✅ 資料導入完成！")


if __name__ == "__main__":
    main()
