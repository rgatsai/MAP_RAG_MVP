"""
Seed script: pre-populate ChromaDB with mock historical data.
Run directly: python seed_data.py
"""

import data_fetcher
import rag_store


def main():
    print("🌱 Seeding ChromaDB with 7 days of mock data...")

    # Generate 7 days of historical weather + business data
    docs = data_fetcher.generate_historical_data("Taipei", days=7)
    print(f"   Generated {len(docs)} documents")

    # Store in ChromaDB
    count = rag_store.add_documents(docs)
    print(f"   Stored {count} documents in ChromaDB")

    # Verify
    info = rag_store.get_all_documents(limit=1)
    print(f"   Total documents in collection: {info['total']}")
    print("✅ Seeding complete!")


if __name__ == "__main__":
    main()
