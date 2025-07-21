from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from core_utils import row_to_document, preprocessing

import os
from tqdm import tqdm
import time

VECTOR_DB_PATH = "chroma_db"
RAW_CSV = "Orders.csv"
CLEANED_PARQUET = "Orders_Cleaned.parquet"

if os.path.exists(VECTOR_DB_PATH):
    print(f"⚠️ Vector store already exists at {VECTOR_DB_PATH}. Delete it to regenerate.")
    exit()

print("📦 Starting preprocessing...")
df = preprocessing(RAW_CSV, CLEANED_PARQUET)
print(f"✅ Preprocessing done. Total records: {len(df)}")

print("🧠 Converting rows to documents...")
documents = [row_to_document(row) for _, row in tqdm(df.iterrows(), total=len(df))]

print("🔗 Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')

print("🧠 Creating vector store...")
start = time.time()
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=VECTOR_DB_PATH,
)
print(f"✅ Vector store created in {time.time() - start:.2f} seconds with {len(documents)} documents.")