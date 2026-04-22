import os
from src.utils.chatbot_utils import BuildChatbot
from src.utils.logger import logging
from src.utils.exception import Custom_exception

from flask import Flask, request, render_template, jsonify, abort

from linebot.v3 import WebhookHandler
#from linebot.v3.webhook import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage as LineTextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError
from dotenv import load_dotenv
load_dotenv()


# initializing flask app
app = Flask(__name__)

# LINE OA
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
 
line_handler = WebhookHandler(LINE_CHANNEL_SECRET)
line_configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

# setting up the chatbot(retriever)
utils = BuildChatbot()
chatbot = utils.initialize_chatbot()



# route for home page
@app.route('/')
def home():
    return render_template('home_page.html')



@app.route('/chat', methods=["GET", "POST"])
def chat():
    data = request.get_json()
    question = data.get('input', '')
    logging.info(f"User Input: {question}")

    config = {"configurable": {"session_id": "chat_1"}}

    response = chatbot.invoke({"input": question},
                              config=config) 

    logging.info(f"Chatbot Response: {response['answer']}")

    return jsonify({"response": response['answer']})


#Line Webhook route
@app.route("/callback", methods=['POST'])
def callback():
    """
    LINE Platform จะ POST มาที่ /callback ทุกครั้งที่มีผู้ใช้ส่งข้อความ
    ต้อง verify signature ก่อนจึงจะประมวลผล event ได้
    """
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    logging.info(f"LINE Webhook received: {body}")
 
    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        logging.error("Invalid LINE signature — request rejected")
        abort(400)
    return 'OK'

@line_handler.add(MessageEvent, message=TextMessageContent)
def handle_line_message(event):
    """
    รับข้อความจากผู้ใช้ LINE -> ส่งเข้า RAG chatbot -> ตอบกลับ
    ใช้ LINE user_id เป็น session_id เพื่อแยก conversation history ต่อ user
    """                        
    user_id = event.source.user_id
    user_message = event.message.text
    logging.info(f"LINE [{user_id}]: {user_message}")
 
    try:
        config = {"configurable": {"session_id": user_id}}
        response = chatbot.invoke({"input": user_message}, config=config)
        reply_text = response['answer']
    except Exception as e:
        logging.error(f"Chatbot error for user {user_id}: {str(e)}")
        reply_text = "ขออภัย เกิดข้อผิดพลาดบางอย่าง กรุณาลองใหม่อีกครั้งนะคะ"
 
    logging.info(f"Reply to [{user_id}]: {reply_text}")
 
    with ApiClient(line_configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[LineTextMessage(text=reply_text)],
            )
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
