from src.components.vectorstore_builder import VectorStoreBuilder
from src.components.chatbot_builder import ChatbotBuilder

# 1. เชื่อมต่อ Vector Store (Pinecone)
vector_builder = VectorStoreBuilder()
# โหลด Vector Store ที่มีอยู่แล้ว (ไม่ต้องรัน pipeline ใหม่ถ้าข้อมูลขึ้นไปแล้ว)
# แต่ถ้าจะเอาชัวร์ ให้ใช้ vector_store จากการรัน pipeline
vector_store = vector_builder.run_pipeline()

#เพิ่ม print debug
print("\n===== TOTAL DOCS IN VECTORSTORE =====")
print(len(vector_store.similarity_search("", k=10)))
print("====================================\n")

# 2. สร้างระบบ Chatbot
bot_builder = ChatbotBuilder()
chat_chain = bot_builder.build_chatbot(vector_store)

# 3. ลองถามคำถาม
question = "มีเสื้อโปโลแนะนำไหมครับ?"

#เพิ่มก่อนเรียก chain
docs = vector_store.similarity_search(question, k=5)

print("\n===== DEBUG DOCS =====")
for i, d in enumerate(docs):
    print(f"{i+1}. {d.page_content}")
print("=====================\n")

response = chat_chain.invoke({
    "input": question, 
    "chat_history": []  # เพิ่มบรรทัดนี้เพื่อบอกว่าตอนนี้ยังไม่มีประวัติการคุย
})

print(f"คำถาม: {question}")
print(f"คำตอบ: {response['answer']}")