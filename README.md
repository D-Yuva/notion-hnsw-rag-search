# notion-hnsw-rag-search

A simple RAG (Retrieval-Augmented Generation) style search tool for Notion pages using HNSWlib for fast vector search.
This script extracts content from your Notion workspace, chunks it, generates embeddings, and builds an HNSW index for efficient semantic search.

⸻

# 🚀 Features

    •	Automatic Notion data extraction using the Notion API
    •	Chunking of long content for better retrieval
    •	SentenceTransformer embeddings (intfloat/e5-base-v2 by default)
    •	HNSWlib vector index for fast and accurate search
    •	Interactive CLI search returning top-matching results with similarity scores

⸻

# 📦 Installation

## Clone the repo

git clone https://github.com/D-Yuva/notion-hnsw-rag-search.git
cd notion-hnsw-rag-search

## Create a virtual environment

python -m venv env
source env/bin/activate # macOS/Linux
env\Scripts\activate # Windows

## Install dependencies

pip install -r requirements.txt

⸻

# 🔑 Setup

    1.	Create a .env file in the project root:
        NOTION_TOKEN=your_notion_integration_token

Note: You can create a Notion Integration and get your token from
https://www.notion.so/my-integrations
Make sure to share your page or database with that integration in Notion.

⸻

# ⚡ Usage

### Step 1 — Build the Index

python src/main.py

    •	Fetches Notion pages
    •	Chunks them into smaller sections
    •	Generates embeddings
    •	Saves:
    •	chunk_vecs.npy
    •	meta.jsonl
    •	skipped_blocks.jsonl
    •	notion_hnsw_hnswlib.index

(All stored inside the /artifacts folder)

#### Step 2 — Search the Index

python src/search.py

    •	Enter your search query
    •	Returns top matches with similarity score, chunk number, and URL

# Configuration

You can adjust these values in src/main.py:
HNSW_M = 32 # Number of bi-directional links per node
EF_CONSTRUCTION = 200 # Controls quality of index building
EF_SEARCH = 128 # Controls recall at query time
MODEL_NAME = "intfloat/e5-base-v2"

For better accuracy (slower & larger), try:
MODEL_NAME = "intfloat/e5-large-v2"

# 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.
