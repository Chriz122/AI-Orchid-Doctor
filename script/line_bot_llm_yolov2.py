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
try:
    model_path = "models\best_sick_5.v7i.v11l.onnx" 
    if not os.path.exists(model_path):
        print(f"在 {model_path} 找不到 YOLO 模型，已停用 YOLO 功能。")
        model = None
    else:
        model = YOLO(model_path)
        print("YOLO 模型載入成功。")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# --- RAG 文件設定 ---
# 使用 PREDEFINED_FILES 列表來指定多個文件
PREDEFINED_FILES = [
    r"static\Anthracnose.docx",
    r"static\Erwinia.docx",
    r"static\Fusarium.docx",
    r"static\Phytophthora.docx",
]

# 文件路徑為 static 資料夾
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_ROOT, 'static')

cc = OpenCC('s2t')  # 將簡體中文轉換成繁體中文
load_dotenv()

# RAG 初始化
def load_document_for_rag(file_path, file_name_for_extension):
    # 根據檔案副檔名載入文件，並回傳相對應的文件載入器
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

def setup_vectorstore_for_rag(all_doc_chunks): 
    # 接受所有文件的分段，並初始化向量資料庫
    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # 初始化 FAISS 向量資料庫
    vectorstore = FAISS.from_documents(all_doc_chunks, embeddings)
    return vectorstore

# 初始化 RAG 系統
rag_chain = None
rag_llm = None
rag_retriever = None
all_documents_for_rag = [] # 儲存所有文件的內容
all_doc_chunks = [] # 儲存所有文件的分段

try:
    print("開始初始化 RAG 系統…")
    for file_name in PREDEFINED_FILES:
        file_path = os.path.join(DATA_FOLDER, file_name) # 構造完整路徑
        print(f"嘗試載入 RAG 文件：{file_path}")
        if not os.path.exists(file_path):
            print(f"RAG 文件不存在: {file_path}")
            continue  # 跳過此文件

        doc_loader, error_msg = load_document_for_rag(file_path, file_name)
        if error_msg:
            print(f"RAG 文件 '{file_name}' 載入失敗: {error_msg}")
        else:
            try:
                documents = doc_loader.load()
                all_documents_for_rag.extend(documents)
                print(f"成功載入並加入文件 '{file_name}'，目前載入總數：{len(all_documents_for_rag)}")
            except Exception as e_load:
                print(f"Error loading documents from '{file_name}': {e_load}")

    if not all_documents_for_rag:
        print("沒有成功載入任何 RAG 文件。RAG 系統將無法運作。")
    else:
        print(f"所有文件載入總數：{len(all_documents_for_rag)}")
        # 新增：使用 RecursiveCharacterTextSplitter 將所有文件分段
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_doc_chunks = text_splitter.split_documents(all_documents_for_rag)
        print(f"文件分段總數：{len(all_doc_chunks)}")
        print("向量資料庫建立完成。")

        vectorstore = setup_vectorstore_for_rag(all_doc_chunks)
        print("Vector store created.")

        rag_llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5"),
            verbose=False,
        )
        print("Ollama LLM 已初始化。")

        rag_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
        print("檢索器已建立。")

        global_memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=rag_llm,
            retriever=rag_retriever,
            memory=global_memory, # 全域記憶（預設行為），但使用者專屬記憶更有助於對話上下文
            chain_type="stuff",
            verbose=False,
        )
        print("RAG 系統初始化完成！")
except Exception as e:
    print(f"RAG 初始化失敗: {e}")
    import traceback
    traceback.print_exc()


def analyze_image_with_yolo(image_path):
    # 根據 YOLO 模型分析圖片並回傳預測結果、標籤以及 RAG 查詢答案
    if not model: 
        # 檢查是否已載入 YOLO 模型
        return image_path, ["YOLO模型未載入"], "YOLO模型未載入，無法進行圖片分析。"

    results = model(image_path)
    result = results[0]
    save_path = image_path.replace(".jpg", "_pred.jpg") # 確保副檔名為 .jpg
    result.save(filename=save_path)
    labels = [model.names[int(cls)] for cls in result.boxes.cls]

    # 基於檢測到的標籤查詢 RAG（這裡使用全局 rag_chain 作為範例）
    # 若需要使用者專屬上下文，則需呼叫 get_user_memory(user_id)
    rag_answer = None
    if rag_chain and labels: 
        try:
            # 為快速測試，使用全局 rag_chain進行查詢
            unique_labels = list(set(labels))
            answers = []
            for label in unique_labels:
                question = f"從文件中找出關於'{label}'的相關資訊。"
                response = rag_chain.invoke({"question": question})
                answer = cc.convert(response.get("answer", "未能獲取答案。"))
                answers.append(f"【{label}】\n{answer}")
            rag_answer = "\n\n".join(answers)
        except Exception as e:
            rag_answer = f"查詢文件時發生錯誤: {e}"
    elif not rag_chain:
        rag_answer = "RAG系統未成功初始化，無法查詢文件。"
    return save_path, labels, rag_answer

# --- 多用戶記憶管理 ---
user_memories = {}
user_last_active = {}
MEMORY_TIMEOUT = timedelta(minutes=10) # 記憶逾時設定

def get_user_memory(user_id):
    # 確保 RAG 元件已初始化
    if not rag_llm or not rag_retriever: 
        print("RAG LLM 或 Retriever 未初始化。無法建立使用者記憶鏈。")
        return None

    now = datetime.now()
    if user_id in user_last_active and now - user_last_active[user_id] > MEMORY_TIMEOUT:
        print(f"使用者 {user_id} 的記憶逾時，正在清除記憶。")
        # 使用者記憶逾時，正在清除記憶。
        user_memories.pop(user_id, None)
        user_last_active.pop(user_id, None)
    user_last_active[user_id] = now
    if user_id not in user_memories:
        print(f"正在為使用者 {user_id} 建立新記憶。")
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
    print(f"Cleared memory for user {user_id}")

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("無效的簽章，請檢查您的 channel secret/access token")
        abort(400)
    except Exception as e:
        # 處理請求時發生錯誤
        app.logger.error(f"Error handling request: {e}")
        abort(500)
    return 'OK'

@handler.add(MessageEvent)
def handle_message(event):
    user_id = event.source.user_id if hasattr(event.source, 'user_id') else 'default_user'
    
    # 處理圖片訊息
    if isinstance(event.message, ImageMessage):
        if not model: 
            # 檢查 YOLO 模型是否可用
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="抱歉，圖片辨識功能目前無法使用。")
            )
            return

        message_id = event.message.id
        static_dir = 'run'
        if not os.path.exists(static_dir):
            # 確保 'run' 目錄存在以儲存臨時圖片
            os.makedirs(static_dir)
        
        image_path = os.path.join(static_dir, f'temp_{message_id}.jpg')

        try:
            message_content = line_bot_api.get_message_content(message_id)
            with open(image_path, 'wb') as f:
                for chunk in message_content.iter_content():
                    f.write(chunk)
            
            pred_path, labels, rag_answer_from_yolo = analyze_image_with_yolo(image_path)
            
            # 組合 YOLO 與 RAG 結果來準備回覆訊息
            reply_messages = []
            yolo_result_text = "偵測結果：" + ", ".join(labels) if labels else "未偵測到任何物體"
            if rag_answer_from_yolo:
                yolo_result_text += f"\n\n【相關資訊】\n{rag_answer_from_yolo}"
            
            reply_messages.append(TextSendMessage(text=yolo_result_text))

            # 構造線上圖片 URL 提供給 LINE 訊息
            # 重要提示：請替換為您實際的 ngrok 或公開 URL，必須能讓 LINE 服務器存取
            base_url = os.getenv("PUBLIC_URL", "YOUR_NGROK_OR_PUBLIC_URL")
            if not base_url.startswith("https://"):
                 print("警告：PUBLIC_URL 應以 https:// 開頭，以確保 LINE ImageSendMessage 正常運作。")

            # 使用檔名部分建立 URL
            pred_filename = os.path.basename(pred_path)
            image_url = f"{base_url}/images/{pred_filename}"
            
            reply_messages.append(
                ImageSendMessage(
                    original_content_url=image_url,
                    preview_image_url=image_url
                )
            )
            line_bot_api.reply_message(event.reply_token, reply_messages)

        except Exception as e:
            app.logger.error(f"Error processing image message: {e}")
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=f"處理圖片時發生錯誤: {e}")
            )
        finally:
            # 延遲刪除臨時檔案
            def delayed_remove(paths, delay):
                def remove_files():
                    time.sleep(delay)
                    for p in paths:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                                print(f"成功刪除檔案: {p}")
                        except Exception as e_remove:
                            print(f"刪除檔案失敗: {p}, 錯誤: {e_remove}")
                threading.Thread(target=remove_files, daemon=True).start()
            
            abs_image_path = os.path.abspath(image_path)
            abs_pred_path = os.path.abspath(pred_path)
            delayed_remove([abs_image_path, abs_pred_path], 180) # 延遲 3 分鐘

    # 處理文字訊息
    elif hasattr(event.message, 'text') and isinstance(event.message.text, str):
        text = event.message.text.strip()
        
        if text == "/清除紀錄":
            clear_user_memory(user_id)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="已清除您的對話紀錄！"))
            return

        rag_answer = None
        if not rag_chain: 
            # 若 RAG 系統未初始化，回覆提示訊息
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="抱歉，文件問答系統目前無法使用。")
            )
            return
            
        try:
            chain = get_user_memory(user_id)
            if not chain:
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text="抱歉，暫時無法處理您的請求，請稍後再試。")
                )
                return

            response = chain.invoke({"question": text})
            rag_answer = cc.convert(response.get("answer", "未能獲取答案。"))
        except Exception as e:
            app.logger.error(f"Error querying RAG chain: {e}")
            rag_answer = f"查詢時發生錯誤，請稍後再試。"
        
        reply_text = rag_answer or "未能獲取答案，或 RAG 系統未就緒。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route("/images/<filename>")
def serve_image(filename):
    # 從 'run' 目錄提供圖片
    static_dir_abs = os.path.abspath('run')
    image_abs_path = os.path.join(static_dir_abs, filename)

    # 安全檢查：確保路徑位於 run 目錄內
    if not os.path.commonpath([static_dir_abs, image_abs_path]) == static_dir_abs:
        app.logger.warning(f"嘗試存取位於 run 目錄之外的檔案: {filename}")
        return "Access denied", 403
        
    if not os.path.exists(image_abs_path):
        app.logger.error(f"找不到圖片檔案: {image_abs_path}")
        return f"File not found: {filename}", 404
    
    print(f"供應圖片: {image_abs_path}")
    return send_file(image_abs_path, mimetype='image/jpeg')

if __name__ == "__main__":
    # 啟動時同時確保 'run' 目錄存在
    if not os.path.exists('run'):
        os.makedirs('run')
    # 本地開發時建議使用 waitress 或 gunicorn 等伺服器；測試時暫用 Flask 內建伺服器
    # 使用 0.0.0.0 可使網路中其他裝置存取（例如，透過 ngrok）
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))