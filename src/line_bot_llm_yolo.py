from flask import Flask, request, abort, send_file
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage

from ultralytics import YOLO
import os
from dotenv import load_dotenv
from opencc import OpenCC
# Langchain RAG 相關 import
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import threading, time
from datetime import datetime, timedelta

# --- LINE Bot 設定 ---
LINE_CHANNEL_SECRET = 'your_channel_secret'
LINE_CHANNEL_ACCESS_TOKEN = 'your_channel_access_token'

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# --- 載入 YOLO 模型 ---
model = YOLO(r"models\best_sick_5.v7i.v11l.onnx")

# --- RAG 文件設定 ---
PREDEFINED_FILE_NAME = r"knowledge\蘭花病徵與診療方式(4).docx"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREDEFINED_FILE_PATH = os.path.join(PROJECT_ROOT, 'knowledge')
cc = OpenCC('s2t')  # 將簡體中文轉換成繁體中文
load_dotenv()

# --- RAG 初始化函數 ---
def load_document_for_rag(file_path, file_name_for_extension):
    # 根據檔案副檔名選擇適當的文件讀取器 (loader)
    file_extension = file_name_for_extension.split(".")[-1]
    if file_extension.lower() == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension.lower() == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_extension.lower() == "txt":
        loader = UnstructuredFileLoader(file_path)
    else:
        return None, f"Unsupported file type: {file_extension}"
    return loader, None

def setup_vectorstore_for_rag(loader):
    # 建立向量資料庫並進行文件切分，用於後續的 RAG 查詢
    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, length_function=len)
    documents = loader.load()
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# --- 啟動時初始化 RAG 系統 ---
rag_chain = None
rag_llm = None
rag_retriever = None
try:
    doc_loader, error_msg = load_document_for_rag(PREDEFINED_FILE_PATH, PREDEFINED_FILE_NAME)
    if error_msg:
        print(f"RAG 文件載入失敗: {error_msg}")
    else:
        vectorstore = setup_vectorstore_for_rag(doc_loader)
        rag_llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            # model=os.getenv("OLLAMA_MODEL", "qwen2.5"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct"),
            verbose=False,
        )
        rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
        memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=rag_llm,
            retriever=rag_retriever,
            memory=memory,
            chain_type="stuff",
            verbose=False,
        )
        print("RAG 系統初始化完成！")
except Exception as e:
    print(f"RAG 初始化失敗: {e}")

def analyze_image_with_yolo(image_path):
    # 使用 YOLO 模型對圖片進行物件分析，並嘗試從 RAG 文件中查詢相關資訊
    results = model(image_path, device=0)
    result = results[0]
    save_path = image_path.replace(".jpg", "_pred.jpg")
    result.save(filename=save_path)
    chinese_labels = ["炭疽", "軟腐病", "黃葉病", "疫病"]
    labels = [chinese_labels[int(cls)] for cls in result.boxes.cls]
    # labels = [model.names[int(cls)] for cls in result.boxes.cls]
    # 新增：自動查詢 RAG 文件
    rag_answer = None
    if rag_chain and labels:
        try:
            question = f"從文件中找出關於'{labels[0]}'的相關資訊。"
            response = rag_chain.invoke({"question": question})
            rag_answer = cc.convert(response.get("answer", "未能獲取答案。"))
        except Exception as e:
            rag_answer = f"查詢文件時發生錯誤: {e}"
    return save_path, labels, rag_answer

# --- 多用戶記憶管理 ---
user_memories = {}
user_last_active = {}
MEMORY_TIMEOUT = timedelta(minutes=10)

def get_user_memory(user_id):
    # 管理使用者互動記憶，如超過10分鐘未互動則重置記憶
    now = datetime.now()
    # 超過 10 分鐘未互動則重置
    if user_id in user_last_active and now - user_last_active[user_id] > MEMORY_TIMEOUT:
        user_memories.pop(user_id, None)
    user_last_active[user_id] = now
    if user_id not in user_memories:
        memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=rag_llm,
            retriever=rag_retriever,
            memory=memory,
            chain_type="stuff",
            verbose=False,
        )
        user_memories[user_id] = chain
    return user_memories[user_id]

def clear_user_memory(user_id):
    user_memories.pop(user_id, None)
    user_last_active.pop(user_id, None)

@app.route("/callback", methods=['POST'])
def callback():
    # 處理來自 LINE 的回呼請求，驗證簽章後再進行處理
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent)
def handle_message(event):
    user_id = event.source.user_id if hasattr(event.source, 'user_id') else 'default'
    # 處理圖片訊息
    if isinstance(event.message, ImageMessage):
        # 設定 run 資料夾並確保存在
        run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
        os.makedirs(run_dir, exist_ok=True)
        message_id = event.message.id
        image_content = line_bot_api.get_message_content(message_id)
        image_path = os.path.join(run_dir, f'temp_{message_id}.jpg')  # 儲存於 run 資料夾
        with open(image_path, 'wb') as f:
            for chunk in image_content.iter_content():
                f.write(chunk)
        pred_path, labels, _ = analyze_image_with_yolo(image_path)
        # 以 YOLO 偵測到的所有不同 label 作為查詢
        rag_answer = None
        if rag_chain and labels:
            try:
                chain = get_user_memory(user_id)
                unique_labels = list(set(labels))
                answers = []
                for label in unique_labels:
                    question = f"從文件中找出關於'{label}'的相關資訊。"
                    response = chain.invoke({"question": question})
                    answer = cc.convert(response.get("answer", "未能獲取答案。"))
                    answers.append(f"【{label}】\n{answer}")
                rag_answer = "\n\n".join(answers)
            except Exception as e:
                rag_answer = f"查詢文件時發生錯誤: {e}"
        reply_text = "偵測結果：" + ", ".join(labels) if labels else "未偵測到任何物體"
        if rag_answer:
            reply_text += f"\n\n【文件查詢結果】\n{rag_answer}"
        line_bot_api.reply_message(
            event.reply_token,
            [
                TextSendMessage(text=reply_text),
                ImageSendMessage(
                    original_content_url=f"https://26e2-140-130-89-129.ngrok-free.app/run/{os.path.basename(pred_path)}",
                    preview_image_url=f"https://26e2-140-130-89-129.ngrok-free.app/run/{os.path.basename(pred_path)}"
                )
            ]
        )
        def delayed_remove(paths, delay):
            # 延遲指定時間後刪除檔案，避免立刻刪除造成衝突或使用問題
            def remove_files():
                time.sleep(delay)
                for p in paths:
                    try:
                        os.remove(p)
                    except Exception as e:
                        print(f"刪除檔案失敗: {p}, 錯誤: {e}")
            threading.Thread(target=remove_files, daemon=True).start()
        delayed_remove([image_path, pred_path], 180) # 延遲 3 分鐘後刪除圖片
    # 處理文字訊息
    elif hasattr(event.message, 'text'):
        text = event.message.text.strip()
        if text == "/清除紀錄":
            clear_user_memory(user_id)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="已清除您的對話紀錄！"))
            return
        rag_answer = None
        if rag_chain:
            try:
                chain = get_user_memory(user_id)
                response = chain.invoke({"question": text})
                rag_answer = cc.convert(response.get("answer", "未能獲取答案。"))
            except Exception as e:
                rag_answer = f"查詢文件時發生錯誤: {e}"
        reply_text = rag_answer or "未能獲取答案。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/run/<filename>")
def serve_image(filename):
    abs_path = os.path.abspath(filename)
    if not os.path.exists(abs_path):
        return f"File not found: {abs_path}", 404
    return send_file(abs_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run()
