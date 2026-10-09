# 🛒 Smart Retail Shelf Monitoring System

### AI-Powered Retail Shelf Detection & Stock Analysis

The **Smart Retail Shelf Monitoring System** is an AI-based computer vision application that automatically analyzes retail shelf images using **YOLO and Streamlit**.

It detects products, identifies SKUs, detects empty shelf spaces, and provides an overall stock status to help monitor shelf availability and restocking needs.

## ✨ Features

* 🛒 **Product Detection** — Detects and counts products on shelves.
* 🏷️ **SKU Detection** — Identifies individual SKU types.
* 🕳️ **Empty-Space Detection** — Detects empty areas on shelves.
* 📊 **Shelf Overview** — Provides total products, product types, and empty spaces.
* 📦 **Inventory Intelligence** — Displays product and SKU stock information.
* 📋 **Final Shelf Analysis** — Provides an overall shelf status: **FULL, LOW, or EMPTY**.
* 🖥️ **Interactive Dashboard** — Built with Streamlit for simple and easy analysis.

## 🤖 AI Models

The system uses three YOLO models:

| Model             | Purpose               |
| ----------------- | --------------------- |
| `product_best.pt` | Product detection     |
| `sku_best.pt`     | SKU detection         |
| `void_best.pt`    | Empty-space detection |

## 🔄 Workflow

```text
📷 Upload Shelf Image
        ↓
🤖 AI Detection
        ↓
🛒 Products + 🏷️ SKUs + 🕳️ Empty Spaces
        ↓
📊 Shelf Overview
        ↓
📦 Inventory Intelligence
        ↓
📋 Final Shelf Analysis
```

## 🧠 Technologies

* Python
* YOLO / Ultralytics
* OpenCV
* NumPy
* Pandas
* Pillow
* Streamlit

## 📁 Project Structure

```text
Smart-Retail-Shelf-Monitoring-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── product_best.pt
│   ├── sku_best.pt
│   └── void_best.pt
│
└── src/
    ├── predict.py
    └── shelf_analysis.py
```

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## 🚀 Future Enhancements

* Real-time CCTV shelf monitoring
* Automatic low-stock alerts
* Inventory history
* Barcode integration
* Multi-store monitoring
* AI-based demand prediction

## 👩‍💻 Developed By

**Amlan Pruthijeet**

### Academic Mini Project

**Domain:** Artificial Intelligence & Computer Vision
