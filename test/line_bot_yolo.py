from flask import Flask, request, abort, send_file
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage

from ultralytics import YOLO
import os

# LINE Bot 設定
LINE_CHANNEL_SECRET = 'your_channel_secret'
LINE_CHANNEL_ACCESS_TOKEN = 'your_channel_access_token'

app = Flask(__name__)
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 初始化 YOLO 模型
model = YOLO(r"models\best_sick_5.v7i.v11l.onnx")

def analyze_image_with_yolo(image_path):
    # 使用 YOLO 模型進行影像分析
    results = model(image_path)
    result = results[0]
    save_path = image_path.replace(".jpg", "_pred.jpg")
    result.save(filename=save_path)  # 儲存預測結果圖片
    labels = [model.names[int(cls)] for cls in result.boxes.cls]  # 提取預測標籤
    return save_path, labels

@app.route("/callback", methods=['POST'])
def callback():
    # 取得 webhook 請求之簽章與內容
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        # 處理 webhook 事件並驗證簽章
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)  # 簽章驗證失敗回傳 400
    return 'OK'

@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # 取得圖片訊息 ID 並下載圖片內容
    message_id = event.message.id
    image_content = line_bot_api.get_message_content(message_id)

    # 儲存下載的圖片檔案
    image_path = f'temp_{message_id}.jpg'
    with open(image_path, 'wb') as f:
        for chunk in image_content.iter_content():
            f.write(chunk)

    # 分析圖片並獲取預測結果及標籤
    pred_path, labels = analyze_image_with_yolo(image_path)

    # 根據預測結果產生回覆訊息
    reply_text = "偵測結果：" + ", ".join(labels) if labels else "未偵測到任何物體"
    line_bot_api.reply_message(
        event.reply_token,
        [
            TextSendMessage(text=reply_text),
            ImageSendMessage(
                original_content_url=f"https://e83f-140-130-89-129.ngrok-free.app/{os.path.basename(pred_path)}",
                preview_image_url=f"https://e83f-140-130-89-129.ngrok-free.app/run/{os.path.basename(pred_path)}"
            )
        ]
    )

    # 設定延遲 30 秒後刪除原始及預測圖片
    import threading, time
    def delayed_remove(paths, delay=30):
        def remove_files():
            time.sleep(delay)
            for p in paths:
                try:
                    os.remove(p)  # 刪除檔案
                except Exception as e:
                    print(f"刪除檔案失敗: {p}, 錯誤: {e}")
        threading.Thread(target=remove_files, daemon=True).start()
    delayed_remove([image_path, pred_path], delay=30)

@app.route("/run/<filename>")
def serve_image(filename):
    # 提供預測圖片下載服務
    abs_path = os.path.abspath(filename)
    if not os.path.exists(abs_path):
        return f"File not found: {abs_path}", 404
    return send_file(abs_path, mimetype='image/jpeg')

if __name__ == "__main__":
    # 啟動 Flask 伺服器
    app.run()
