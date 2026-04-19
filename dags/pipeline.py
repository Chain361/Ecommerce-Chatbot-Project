from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
import sys
import os

# เพิ่ม Path เพื่อให้ Airflow หา Module ใน src เจอ
sys.path.append('/opt/airflow')

from src.components.data_collection import DataCollection
from src.components.data_cleaning import DataCleaner
from src.components.vectorstore_builder import VectorStoreBuilder
from src.components.chatbot_builder import ChatbotBuilder
from langchain_pinecone import PineconeVectorStore

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 17), # หรือปรับเป็นวันที่ปัจจุบัน
    'retries': 1,
    'retry_delay': timedelta(minutes=4),
}

dag = DAG(
    'Ecommerce-Chatbot-Pipeline',
    default_args=default_args,
    description='Ecommerce Chatbot Pipeline for Thai Products',
    schedule_interval=None,
    catchup=False,
)

# --- Task Functions ---

def collect_data():
    logging.info("Starting Data Collection Task")
    DataCollection().initiate_data_collection()

def clean_data():
    logging.info("Starting Data Cleaning Task")
    DataCleaner().clean_data()

def build_vectorstore():
    logging.info("Starting VectorStore Build Task")
    # ตัวนี้จะใช้ index_name="rough-v2" ตาม default ใน class
    VectorStoreBuilder().run_pipeline()       

def build_chatbot():
    logging.info("Starting Chatbot Build Task")
    builder_config = VectorStoreBuilder()
    embeddings = builder_config.create_embeddings()
    
    # แก้ไข index_name ให้ตรงกับ rough-v2 ที่อยู่ใน Pinecone และ VectorStoreBuilder
    index_name = "rough-v2" 
    
    logging.info(f"Loading existing index: {index_name}")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
   
    # ส่ง vector_store เข้าไปสร้าง Chain ใน ChatbotBuilder
    chatbot = ChatbotBuilder()
    chatbot.build_chatbot(vector_store)
    logging.info("Chatbot Chain built successfully")


# --- DAG Structure ---

with dag:
    task1 = PythonOperator(
        task_id='data_collection',
        python_callable=collect_data
    )

    task2 = PythonOperator(
        task_id='data_cleaning',
        python_callable=clean_data
    )

    task3 = PythonOperator(
        task_id='vectorstore_build',
        python_callable=build_vectorstore
    )

    task4 = PythonOperator(
        task_id='chatbot_build',
        python_callable=build_chatbot
    )

    # กำหนดลำดับการทำงาน
    task1 >> task2 >> task3 >> task4