import os
import sys
from dotenv import load_dotenv
from opencc import OpenCC
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
from ultralytics import YOLO
import cv2
import torch
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QFileDialog, QTextEdit, QScrollArea, QMessageBox, QLineEdit,
    QGroupBox
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtCore import Qt, Slot

# --- 全域設定 ---
load_dotenv()
cc = OpenCC('s2t')  # 將簡體中文轉換成繁體中文

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Langchain RAG 配置 ---
PREDEFINED_FILE_NAME = r"knowledge\蘭花病徵與診療方式.docx"
PREDEFINED_FILE_PATH = os.path.join(SCRIPT_DIR, PREDEFINED_FILE_NAME)

# --- YOLO 配置 ---
YOLO_MODEL_PATH = r'models\best_sick_5.v7i.v11l.onnx' # 更新此路徑

# --- Langchain RAG 函式 (由 Code 1 修改，假定與合併版本相同) ---
def load_document_for_rag(file_path, file_name_for_extension):
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
        return None, f"不支援的檔案類型：{file_extension}"
    return loader, None

def setup_vectorstore_for_rag(loader, log_callback):
    model_name = "nomic-ai/nomic-embed-text-v2-moe"
    model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        log_callback(f"錯誤：無法載入 HuggingFace Embeddings: {e}")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, length_function=len)
    log_callback("正在載入 RAG 文件...")
    try:
        documents = loader.load()
    except Exception as e:
        log_callback(f"錯誤：無法載入文件內容: {e}")
        return None
    if not documents:
        log_callback("錯誤：文件為空或無法載入。")
        return None
    log_callback(f"文件已載入。頁數/部分數量：{len(documents)}")
    log_callback("正在分割文件為區塊...")
    doc_chunks = text_splitter.split_documents(documents)
    if not doc_chunks:
        log_callback("錯誤：無法將文件分割成區塊。")
        return None
    log_callback(f"文件分割為 {len(doc_chunks)} 個區塊。")
    log_callback("正在建立向量資料庫 (FAISS)...")
    try:
        vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    except Exception as e:
        log_callback(f"錯誤：無法建立 FAISS 向量資料庫: {e}")
        return None
    log_callback("向量資料庫建立成功。")
    return vectorstore

def create_chain_for_rag(vectorstore, log_callback):
    log_callback("正在建立 Ollama LLM 實例...")
    try:
        llm = Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.getenv("OLLAMA_MODEL", "qwen2.5"),
            verbose=False,
        )
        log_callback(f"使用的 Ollama 模型：{os.getenv('OLLAMA_MODEL', 'qwen2.5')}")
    except Exception as e:
        log_callback(f"錯誤：無法建立 Ollama LLM 實例: {e}")
        return None
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})
    memory = ConversationBufferMemory(output_key="answer", memory_key="chat_history", return_messages=True)
    log_callback("正在建立 ConversationalRetrievalChain...")
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            verbose=False,
        )
    except Exception as e:
        log_callback(f"錯誤：無法建立 ConversationalRetrievalChain: {e}")
        return None
    log_callback("RAG 鏈建立成功。")
    return chain

# --- 主 PySide6 應用程式視窗 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO + RAG 文件問答與聊天")
        self.setGeometry(100, 100, 1000, 900) # 增加高度以顯示聊天介面

        # --- 類別成員變數 ---
        self.yolo_model = None
        self.rag_chain = None
        self.current_image_path = None
        self.yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # --- 使用者介面元件 ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 上方控制區（與 YOLO 有關）
        yolo_controls_group = QGroupBox("YOLO 圖片辨識與查詢")
        yolo_controls_layout = QVBoxLayout()

        controls_layout_buttons = QHBoxLayout()
        self.btn_select_image = QPushButton("選擇圖片")
        self.btn_select_image.clicked.connect(self.select_image)
        controls_layout_buttons.addWidget(self.btn_select_image)

        self.btn_process = QPushButton("處理圖片並查詢")
        self.btn_process.clicked.connect(self.process_image_and_query_document)
        self.btn_process.setEnabled(False)
        controls_layout_buttons.addWidget(self.btn_process)
        yolo_controls_layout.addLayout(controls_layout_buttons)

        # 圖片顯示區
        image_area_layout = QHBoxLayout()
        self.lbl_selected_image = QLabel("尚未選擇圖片")
        self.lbl_selected_image.setMinimumSize(380, 280) # 調整尺寸
        self.lbl_selected_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_selected_image.setStyleSheet("border: 1px solid gray;")
        image_area_layout.addWidget(self.lbl_selected_image)

        self.lbl_yolo_result_image = QLabel("YOLO 辨識結果")
        self.lbl_yolo_result_image.setMinimumSize(380, 280) # 調整尺寸
        self.lbl_yolo_result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_yolo_result_image.setStyleSheet("border: 1px solid gray;")
        image_area_layout.addWidget(self.lbl_yolo_result_image)
        yolo_controls_layout.addLayout(image_area_layout)

        # YOLO 偵測結果與基於 YOLO 的 RAG 回答
        yolo_results_layout = QHBoxLayout()
        yolo_detections_group_widget = QWidget()
        yolo_v_layout = QVBoxLayout(yolo_detections_group_widget)
        yolo_v_layout.addWidget(QLabel("YOLO 偵測物件："))
        self.txt_yolo_detections = QTextEdit()
        self.txt_yolo_detections.setReadOnly(True)
        self.txt_yolo_detections.setFixedHeight(80) # 調整高度
        yolo_v_layout.addWidget(self.txt_yolo_detections)
        yolo_results_layout.addWidget(yolo_detections_group_widget)

        rag_yolo_answer_group_widget = QWidget()
        rag_yolo_v_layout = QVBoxLayout(rag_yolo_answer_group_widget)
        rag_yolo_v_layout.addWidget(QLabel(f"依據辨識結果查詢 '{PREDEFINED_FILE_NAME}'："))
        self.txt_rag_answer_yolo = QTextEdit() # 為了清晰重新命名
        self.txt_rag_answer_yolo.setReadOnly(True)
        self.txt_rag_answer_yolo.setFont(QFont("Arial", 10)) # 調整字型
        self.txt_rag_answer_yolo.setFixedHeight(100) # 調整高度
        rag_yolo_v_layout.addWidget(self.txt_rag_answer_yolo)
        yolo_results_layout.addWidget(rag_yolo_answer_group_widget)
        yolo_controls_layout.addLayout(yolo_results_layout)
        yolo_controls_group.setLayout(yolo_controls_layout)
        main_layout.addWidget(yolo_controls_group)

        # --- 與文件直接對話區 ---
        chat_group = QGroupBox(f"直接與文件 '{PREDEFINED_FILE_NAME}' 對話")
        chat_layout = QVBoxLayout()

        self.txt_chat_history = QTextEdit()
        self.txt_chat_history.setReadOnly(True)
        self.txt_chat_history.setFont(QFont("Arial", 10))
        self.txt_chat_history.setPlaceholderText("對話歷史將顯示於此...")
        chat_layout.addWidget(self.txt_chat_history) # 占用可用空間

        chat_input_layout = QHBoxLayout()
        self.txt_chat_input = QLineEdit()
        self.txt_chat_input.setPlaceholderText("在這裡輸入您的問題...")
        self.txt_chat_input.returnPressed.connect(self.send_chat_message) # 按 Enter 鍵送出
        chat_input_layout.addWidget(self.txt_chat_input)

        self.btn_send_chat = QPushButton("發送")
        self.btn_send_chat.clicked.connect(self.send_chat_message)
        chat_input_layout.addWidget(self.btn_send_chat)
        chat_layout.addLayout(chat_input_layout)
        chat_group.setLayout(chat_layout)
        main_layout.addWidget(chat_group)

        # 日誌區
        log_group = QGroupBox("日誌")
        log_layout = QVBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(100) # 調整高度
        log_layout.addWidget(self.txt_log)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        # 初始化模型
        self.log_message("應用程式啟動。")
        self._initialize_systems()

    def log_message(self, message):
        self.txt_log.append(message)
        QApplication.processEvents()

    def _initialize_systems(self):
        self.log_message(f"使用的 YOLO 設備：{self.yolo_device}")
        # 1. 初始化 Langchain RAG
        self.log_message("正在初始化 RAG 系統...")
        if not os.path.exists(PREDEFINED_FILE_PATH):
            self.log_message(f"錯誤：預定義文件 '{PREDEFINED_FILE_PATH}' 未找到。")
            QMessageBox.critical(self, "錯誤", f"RAG 文件 '{PREDEFINED_FILE_NAME}' 未找到!")
            return
        doc_loader, error_msg = load_document_for_rag(PREDEFINED_FILE_PATH, PREDEFINED_FILE_NAME)
        if error_msg:
            self.log_message(f"錯誤：{error_msg}")
            QMessageBox.critical(self, "錯誤", f"載入 RAG 文件失敗: {error_msg}")
            return
        if doc_loader:
            vectorstore = setup_vectorstore_for_rag(doc_loader, self.log_message)
            if vectorstore:
                self.rag_chain = create_chain_for_rag(vectorstore, self.log_message)
                if self.rag_chain:
                    self.log_message(f"RAG 系統已成功處理文件 '{PREDEFINED_FILE_NAME}'。")
                    self.btn_send_chat.setEnabled(True) # 啟用聊天送出按鈕
                else:
                    self.log_message("錯誤：RAG 鏈創建失敗。")
                    QMessageBox.warning(self, "警告", "RAG 鏈創建失敗，請檢查日誌。")
                    self.btn_send_chat.setEnabled(False)
            else:
                self.log_message("錯誤：向量資料庫創建失敗。")
                QMessageBox.warning(self, "警告", "RAG 向量資料庫創建失敗。")
                self.btn_send_chat.setEnabled(False)
        else:
            self.log_message(f"錯誤：文件 '{PREDEFINED_FILE_NAME}' 載入失敗。")
            QMessageBox.warning(self, "警告", f"RAG 文件 '{PREDEFINED_FILE_NAME}' 載入失敗。")
            self.btn_send_chat.setEnabled(False)

        # 2. 初始化 YOLO 模型
        self.log_message("正在初始化 YOLO 模型...")
        try:
            if not os.path.exists(YOLO_MODEL_PATH):
                 self.log_message(f"錯誤：YOLO 模型文件 '{YOLO_MODEL_PATH}' 未找到。")
                 QMessageBox.critical(self, "錯誤", f"YOLO 模型 '{YOLO_MODEL_PATH}' 未找到!")
                 return
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            # 進行虛擬預測以確認模型載入
            # 確保虛擬預測的輸入尺寸符合 ONNX 模型預期輸入
            _ = self.yolo_model.predict(torch.zeros(1, 3, 640, 640).to(self.yolo_device), verbose=False, imgsz=640) # 虛擬預測
            self.log_message("YOLO 模型載入成功。")
        except Exception as e:
            self.log_message(f"錯誤：YOLO 模型載入失敗: {e}")
            QMessageBox.critical(self, "錯誤", f"YOLO 模型載入失敗: {e}")
            self.yolo_model = None

        if self.rag_chain and self.yolo_model:
            self.btn_process.setEnabled(True) # 啟用 YOLO 處理按鈕
            self.log_message("所有系統準備就緒。")
        elif self.rag_chain:
            self.log_message("RAG 系統就緒，但 YOLO 未就緒。僅能進行文字對話。")
        else:
            self.log_message("錯誤：系統初始化失敗，功能受限。")

    @Slot()
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片檔案", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if file_path:
            self.current_image_path = file_path
            pixmap = QPixmap(file_path)
            self.lbl_selected_image.setPixmap(
                pixmap.scaled(self.lbl_selected_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            self.log_message(f"已選擇圖片: {file_path}")
            self.lbl_yolo_result_image.setText("等待處理")
            self.txt_yolo_detections.clear()
            self.txt_rag_answer_yolo.clear()

    def _display_yolo_image(self, image_np_bgr):
        try:
            image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = image_np_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(image_np_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.lbl_yolo_result_image.setPixmap(
                 pixmap.scaled(self.lbl_yolo_result_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        except Exception as e:
            self.log_message(f"顯示 YOLO 圖片時發生錯誤: {e}")
            self.lbl_yolo_result_image.setText("無法顯示圖片")

    @Slot()
    def process_image_and_query_document(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "請先選擇一張圖片。")
            return
        if not self.yolo_model:
            QMessageBox.critical(self, "錯誤", "YOLO 模型未載入。")
            return
        if not self.rag_chain:
            QMessageBox.critical(self, "錯誤", "RAG 系統未初始化。")
            return

        self.log_message(f"正在處理圖片: {self.current_image_path}")
        self.txt_yolo_detections.clear()
        self.txt_rag_answer_yolo.clear()
        QApplication.processEvents()

        try:
            results = self.yolo_model.predict(
                source=self.current_image_path,
                conf=0.25,
                imgsz=640, # 確保此尺寸與訓練尺寸匹配或為常用尺寸
                verbose=False
            )
            
            if not results or not results[0].boxes:
                self.log_message("YOLO 未偵測到任何物件。")
                self.txt_yolo_detections.setText("未偵測到物件。")
                self._display_yolo_image(cv2.imread(self.current_image_path))
                return

            img_with_boxes = results[0].plot()
            self._display_yolo_image(img_with_boxes)

            detected_object_names_full = []
            primary_query_term = None
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                detected_object_names_full.append(f"{class_name} (信心度: {conf:.2f})")
                if i == 0:
                    primary_query_term = class_name
            
            self.txt_yolo_detections.setText("\n".join(detected_object_names_full))
            self.log_message(f"YOLO 偵測結果: {', '.join(detected_object_names_full)}")

            if not primary_query_term:
                self.log_message("無主要偵測物件可供查詢。")
                self.txt_rag_answer_yolo.setText("無偵測物件，無法查詢。")
                return

            self.log_message(f"正在使用 '{primary_query_term}' 查詢 RAG 系統...")
            question_to_ask = f"從文件中找出關於'{primary_query_term}'的相關資訊。"
            self.txt_chat_history.append(f"<font color='blue'><b>YOLO 觸發查詢:</b></font> {question_to_ask}\n")

            response = self.rag_chain.invoke({"question": question_to_ask})
            assistant_response = response.get("answer", "未能獲取答案。")
            assistant_response_traditional = cc.convert(assistant_response)
            
            self.txt_rag_answer_yolo.setText(assistant_response_traditional)
            self.log_message(f"RAG (YOLO) 回答: {assistant_response_traditional[:100]}...")
            self.txt_chat_history.append(f"<font color='green'><b>文檔助手:</b></font> {assistant_response_traditional}\n---")

        except Exception as e:
            self.log_message(f"處理過程中發生錯誤: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            QMessageBox.critical(self, "錯誤", f"處理圖片或查詢時發生錯誤: {e}")
            self.txt_rag_answer_yolo.setText(f"查詢錯誤: {e}")

    @Slot()
    def send_chat_message(self):
        user_input = self.txt_chat_input.text().strip()
        if not user_input:
            return # 如果輸入為空則不處理

        if not self.rag_chain:
            QMessageBox.warning(self, "錯誤", "RAG 系統未初始化，無法進行聊天。")
            return

        self.txt_chat_input.clear()
        self.txt_chat_history.append(f"<font color='blue'><b>您:</b></font> {user_input}\n")
        self.log_message(f"使用者提問: {user_input}")
        QApplication.processEvents() # 更新介面

        try:
            self.setCursor(Qt.CursorShape.WaitCursor) # 顯示忙碌游標
            # ConversationalRetrievalChain 的記憶體負責管理 chat_history
            response = self.rag_chain.invoke({"question": user_input})
            assistant_response = response.get("answer", "抱歉，我無法回答這個問題。")
            assistant_response_traditional = cc.convert(assistant_response)

            self.txt_chat_history.append(f"<font color='green'><b>文檔助手:</b></font> {assistant_response_traditional}\n---")
            self.log_message(f"RAG (聊天) 回答: {assistant_response_traditional[:100]}...")

        except Exception as e:
            error_msg = f"與文檔助手對話時發生錯誤: {e}"
            self.log_message(error_msg)
            import traceback
            self.log_message(traceback.format_exc())
            self.txt_chat_history.append(f"<font color='red'><b>系統錯誤:</b></font> {error_msg}\n---")
            QMessageBox.critical(self, "聊天錯誤", error_msg)
        finally:
            self.unsetCursor() # 還原游標
            self.txt_chat_history.verticalScrollBar().setValue(self.txt_chat_history.verticalScrollBar().maximum())

if __name__ == "__main__":
    if not os.path.exists(PREDEFINED_FILE_PATH):
        print(f"嚴重錯誤：預定義文件 '{PREDEFINED_FILE_PATH}' 未找到。")
        sys.exit(1)
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"嚴重錯誤：YOLO 模型 '{YOLO_MODEL_PATH}' 未找到。")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())