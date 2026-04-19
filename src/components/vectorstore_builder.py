import os 
import sys 
import time
import csv # เพิ่มการใช้ csv module
from typing import List
from dataclasses import dataclass

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from src.utils.logger import logging
from src.utils.exception import Custom_exception
from dotenv import load_dotenv

load_dotenv()

@dataclass
class VectorStoreBuilderConfig:
    is_airflow = os.getenv("IS_AIRFLOW", "false").lower() == "true"

    if is_airflow:
        path = "/opt/airflow/artifacts/data_cleaned.csv"
    else:
        path = "artifacts/data_cleaned.csv"

class VectorStoreBuilder:
    def __init__(self):
        self.vectorstore_builder_config = VectorStoreBuilderConfig()
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.nvidia_api_key or not self.pinecone_api_key:
            raise ValueError("Required API keys not set")

    def load_data(self, data_path: str) -> List[Document]:
        try:
            logging.info(f"Loading data from {data_path}")
            docs = []

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"ไม่พบไฟล์ CSV ที่: {data_path}")

            with open(data_path, "r", encoding="utf-8") as f:
                # ใช้ csv.reader แทนการ .split(",") แบบธรรมดา
                reader = csv.reader(f)
                
                # ข้าม Header (บรรทัดแรก)
                header = next(reader, None) 

                for row in reader:
                    # ป้องกันบรรทัดว่างหรือข้อมูลไม่ครบ
                    if not row or len(row) < 4:
                        continue
                    
                    # แกะค่าตามลำดับ: name, brand, price, rating (จาก DataCleaner ที่ไม่มี index)
                    name, brand, price, rating = row[0], row[1], row[2], row[3]

                    # detect category แบบง่าย
                    if "เสื้อโปโล" in name:
                        category = "เสื้อโปโล"
                    elif "เสื้อยืด" in name:
                        category = "เสื้อยืด"
                    elif "กางเกง" in name:
                        category = "กางเกง"
                    else:
                        category = "อื่นๆ"

                    content = f"""
ชื่อสินค้า: {name.strip()}
แบรนด์: {brand.strip()}
ราคา: {price.strip()} บาท
คะแนนรีวิว: {rating.strip()}
ประเภทสินค้า: {category}
                    """.strip()

                    docs.append(Document(page_content=content, metadata={"source": data_path, "category": category}))

            logging.info(f"Formatted {len(docs)} documents successfully.")
            return docs

        except Exception as e:
            logging.error(f"Error in loading data: {str(e)}")
            raise Custom_exception(e, sys)

    def create_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        try: 
            logging.info("Initializing HF BGE Embeddings (Multilingual).")
            embeddings = HuggingFaceEndpointEmbeddings(
                model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            )
            logging.info("Embeddings initialized successfully.")
            return embeddings
        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise Custom_exception(e, sys)

    def test_embeddings(self, embeddings: HuggingFaceEndpointEmbeddings):
        try:
            logging.info("Testing embeddings with sample text...")
            test_text = "ทดสอบระบบค้นหา"
            test_embedding = embeddings.embed_query(test_text)
            logging.info(f"Test embedding dimension: {len(test_embedding)}")
            return True
        except Exception as e:
            logging.error(f"Embeddings test failed: {str(e)}")
            raise Custom_exception(e, sys)

    def create_vector_store(self, documents: List[Document], 
                            embeddings: HuggingFaceEndpointEmbeddings, 
                            index_name: str = 'rough-v2') -> PineconeVectorStore:
        try:
            logging.info(f"Connecting to Pinecone index: {index_name}")
            pc = Pinecone(api_key=self.pinecone_api_key)

            # ตรวจสอบ Dimension (paraphrase-multilingual คือ 384)
            if index_name in pc.list_indexes().names():
                index_info = pc.describe_index(index_name)
                if index_info.dimension != 384:
                    logging.warning(f"Index dimension mismatch. Deleting {index_name}...")
                    pc.delete_index(index_name)
                    time.sleep(10)

            if index_name not in pc.list_indexes().names():
                logging.info(f"Creating new index: {index_name}")
                pc.create_index(name=index_name,
                                 dimension = 384,
                                 metric="cosine",
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                time.sleep(10)

            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            existing_count = stats.get("total_vector_count", 0)

            if existing_count > 0:
                logging.info(f"Index already has {existing_count} vectors. Skipping upload.")
                vector_store = PineconeVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=embeddings
                )
            else:
                logging.info(f"Uploading {len(documents)} documents to Pinecone...")
                vector_store = PineconeVectorStore.from_documents(
                    documents=documents,
                    index_name=index_name,
                    embedding=embeddings
                )

            return vector_store
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            raise Custom_exception(e, sys)

    def run_pipeline(self) -> PineconeVectorStore:
        try:
            logging.info("Starting vectorstore pipeline")
            docs = self.load_data(self.vectorstore_builder_config.path)
            if not docs:
                raise ValueError("No documents were loaded. Pipeline stopped.")
                
            embeddings = self.create_embeddings()
            self.test_embeddings(embeddings)
            vector_store = self.create_vector_store(docs, embeddings)

            logging.info("Vectorstore pipeline completed successfully")
            return vector_store
        except Exception as e:
            logging.error(f"Error in pipeline execution: {str(e)}")
            raise Custom_exception(e, sys)

if __name__=="__main__":
    pipe = VectorStoreBuilder()
    pipe.run_pipeline()