# yolo_LLM_linebot

蘭花 AI 醫生

## 操作範例

[範例操作](assets\sample.jpg)

## 項目結構

```
yolo_LLM_linebot
├─ assets                  # assets 資料夾，存放圖片等資源
│  └─ sample.jpg           # 範例圖片檔案
├─ models                  # models 資料夾，存放模型檔案
│  └─ best_sick_5.v7i.v11l.onnx  # ONNX 模型檔案
├─ run                     # run 資料夾
├─ script                  # script 資料夾，包含執行腳本
│  ├─ line_bot_llm_yolo.py    # Line Bot (LLM + YOLO)
│  └─ line_bot_llm_yolov2.py  # Line Botv2 (LLM + YOLO)
├─ static                     # static 資料夾，存放靜態文件
│  ├─ Anthracnose.docx        # 檔案：Anthracnose
│  ├─ Erwinia.docx            # 檔案：Erwinia
│  ├─ Fusarium.docx           # 檔案：Fusarium
│  ├─ Phytophthora.docx       # 檔案：Phytophthora
│  ├─ 蘭花病徵與診療方式.docx       # 檔案：蘭花病徵與診療方式
│  ├─ 蘭花病徵與診療方式(1).docx    # 檔案：蘭花病徵與診療方式(1)
│  ├─ 蘭花病徵與診療方式(2).docx    # 檔案：蘭花病徵與診療方式(2)
│  ├─ 蘭花病徵與診療方式(3).docx    # 檔案：蘭花病徵與診療方式(3)
│  └─ 蘭花病徵與診療方式(4).docx    # 檔案：蘭花病徵與診療方式(4)
└─ test                       # test 資料夾，存放測試文件
   ├─ line_bot_yolo.py        # 測試：line_bot_yolo.py
   └─ pdf_yolo_PySide6.py     # 測試：pdf_yolo_PySide6.py
```
