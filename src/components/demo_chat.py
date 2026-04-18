from src.components.vectorstore_builder import VectorStoreBuilder
from src.components.chatbot_builder import ChatbotBuilder

# 1. เชื่อมต่อ Vector Store (Pinecone)
vector_builder = VectorStoreBuilder()
# โหลด Vector Store ที่มีอยู่แล้ว (ไม่ต้องรัน pipeline ใหม่ถ้าข้อมูลขึ้นไปแล้ว)
# แต่ถ้าจะเอาชัวร์ ให้ใช้ vector_store จากการรัน pipeline
vector_store = vector_builder.run_pipeline() 

# 2. สร้างระบบ Chatbot
bot_builder = ChatbotBuilder()
chat_chain = bot_builder.build_chatbot(vector_store)

# 3. ลองถามคำถาม
question = "มีเสื้อโปโลแนะนำไหมครับ?"
response = chat_chain.invoke({
    "input": question, 
    "chat_history": []  # เพิ่มบรรทัดนี้เพื่อบอกว่าตอนนี้ยังไม่มีประวัติการคุย
})

print(f"คำถาม: {question}")
print(f"คำตอบ: {response['answer']}")