import os 
import sys
from typing import Any

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_nvidia import NVIDIAEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.utils.logger import logging
from src.utils.exception import Custom_exception
from dotenv import load_dotenv
load_dotenv()



class BuildRetrievalchain:
    """
    contains helper function for creating chatbot
    embeddings, llm, prompt, vector_store, retriever, retrieval_chain
    """

    def __init__(self):
        pass



    # def load_embeddings(self) -> NVIDIAEmbeddings:
    #     try:
    #         logging.info("Initializing NVIDIA Embeddings.")
    #         embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2",
    #                                     api_key=os.getenv("NVIDIA_API_KEY"),
    #                                     truncate="NONE")
                
    #         logging.info("Embeddings initialized successfully.")
    #         return embeddings
            
    #     except Exception as e:
    #         logging.error(f"Error initializing embeddings: {str(e)}")
    #         raise Custom_exception(e, sys)



    def load_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        try: 
            logging.info("Initializing HF BGE Embeddings.")
            embeddings = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-small-en-v1.5",
                huggingfacehub_api_token=os.getenv("HF_API_KEY"),
            )
            logging.info("Embeddings initialized successfully.")
            return embeddings

        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise Custom_exception(e, sys)

        

    def load_llm(self):
        try:
            logging.info("Initializing Llama2 model with Groq")
            llm = ChatGroq(temperature=0.6,
                        model_name="llama-3.3-70b-versatile",
                        #model_name="llama-3.1-8b-instant",
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        max_tokens=4096)
            
            logging.info("LLM initialized successfully")
            return llm
            
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            raise Custom_exception(e, sys)
        


    def setup_prompt(self):
        try:
            logging.info("Creating prompt template")
            system_prompt = """
            คุณคือ AI ผู้ช่วยร้านค้าออนไลน์ (E-commerce Chatbot)

            หน้าที่ของคุณ:
            - ช่วยค้นหาและแนะนำสินค้าในระบบ
            - ใช้ข้อมูลจาก context เท่านั้น
            - ตอบคำถามเกี่ยวกับสินค้า ราคา รีวิว และส่วนลด

            ข้อมูลสินค้าใน context อาจมี:
            {context}

            กฎสำคัญ:
            1. ห้ามสร้างข้อมูลเองโดยเด็ดขาด
            2. ห้ามข้ามประเภทสินค้า
            3. ต้องเลือกสินค้าที่ตรงคำถามที่สุดก่อน
            4. ถ้ามีสินค้าที่เกี่ยวข้อง → ห้ามตอบว่าไม่พบสินค้า
            5. ใช้ข้อมูลจาก context เท่านั้น
            6. ถ้าไม่ตรง 100% ให้เลือก "ใกล้เคียงที่สุด"

            ========================
            รูปแบบการตอบ:
            สินค้าแนะนำ:

            1)
            ชื่อสินค้า: xxx
            แบรนด์: xxx
            ราคา: ฿xxx
            คะแนนรีวิว: x.x

            ========================
            กรณีไม่มีจริงๆ:
            "ขออภัยค่ะ ไม่พบสินค้าที่ตรงกับความต้องการในขณะนี้"

            โทนการตอบ:
            - สุภาพ
            - เป็นกันเอง
            """
        
            prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                        MessagesPlaceholder(variable_name="chat_history"),  # For maintaining conversation history
                                                        ("human", "{input}")]) 
            
            logging.info("Prompt template has been created")
            return prompt
        
        except Exception as e:
            logging.error(f"Error creating prompt: {str(e)}")
            raise Custom_exception(e, sys)
            


    def load_vectorstore(self, embeddings):
        try:
            logging.info("Loading vectorstore ")  
            vector_store = PineconeVectorStore.from_existing_index(index_name="rough",           # ecommerce-chatbot-project
                                                                   embedding=embeddings)

            logging.info("Successfully loaded vectorstore")
            return vector_store
        
        except Exception as e:
            raise Custom_exception(e, sys)
        


    def build_retriever(self, vector_store: PineconeVectorStore):
        try:

            logging.info("Initializing vector_store as retriever")
            retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                                  search_kwargs={"k": 5,                    # Number of documents to return
                                                                 "score_threshold": 0.7})   # Minimum relevance threshold
                                                            
            logging.info("Retriever has be initializing")
            return retriever
        
        except Exception as e:
            logging.info(f"Error initializing retriever: {str(e)}")
            raise Custom_exception(e, sys)
        


    def build_chains(self, llm: Any, prompt: ChatPromptTemplate, retriever: Any):
        try:
            logging.info("Creating stuff document chain...")
            doc_chain = create_stuff_documents_chain(llm=llm, 
                                                    prompt=prompt,
                                                    output_parser=StrOutputParser(),
                                                    document_variable_name="context")
            
            logging.info("Creating retrieval chain...")
            retrieval_chain = create_retrieval_chain(retriever=retriever, 
                                                    combine_docs_chain=doc_chain)
            
            logging.info("Chains created successfully")
            return retrieval_chain
        
        except Exception as e:
            logging.info(f"Error creating chains {str(e)}")
            raise Custom_exception(e, sys)
        


    def build_retrieval_chain(self):
        try:
            embeddings = self.load_embeddings()
            llm = self.load_llm()
            prompt = self.setup_prompt()
            vector_store = self.load_vectorstore(embeddings)
            retriever = self.build_retriever(vector_store)
            retrieval_chain = self.build_chains(llm, prompt, retriever)

            return retrieval_chain
        except Exception as e:
            raise Custom_exception(e, sys)
        
    
    

class BuildChatbot:
    def __init__(self):
        self.store = {}  # Persistent dictionary to maintain chat history


    def get_session_id(self, session_id: str) -> BaseChatMessageHistory:
        """creates and retrieves a chat history session."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]


    def initialize_chatbot(self):
        """Initializes the chatbot with session memory."""
        utils = BuildRetrievalchain()
        retrieval_chain = utils.build_retrieval_chain()

        chatbot = RunnableWithMessageHistory(runnable=retrieval_chain,
                                             get_session_history=self.get_session_id,
                                             input_messages_key="input",
                                             history_messages_key="chat_history",
                                             output_messages_key="answer")

        return chatbot
