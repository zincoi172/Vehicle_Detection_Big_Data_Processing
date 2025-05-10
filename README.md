# 🚗 Smart Vehicle Detection and Parking Occupancy Management

**Course:** Data 228 – Big Data Technologies

**Instructor:** Dr. Guannan Liu

**Team:** Cheng-Huan Yu, Chun-Chieh Kuo, Khac Minh Dai Vo, Lam Tran

## 📌 Overview

This project presents a scalable, real-time system for detecting vehicles and monitoring parking occupancy using deep learning and big data technologies. The system integrates **YOLOv8s** for vehicle detection, **Apache Kafka** for real-time message streaming, and **PySpark** for distributed batch inference. It supports both real-time alerts and historical analytics via PostgreSQL.

## 🧠 Key Features

* **YOLOv8 Model Training**:

  * Pretrained on the COCO 2017 dataset.
  * Fine-tuned on the MIO-TCD dataset for traffic-specific vehicle detection.
* **Kafka-based Real-Time Monitoring**:

  * Live parking spot detection using OpenCV.
  * Telegram bot for real-time alerts (with 30-second stability logic).
  * Snapshot logs stored in PostgreSQL every 3 minutes.
* **Distributed Inference with PySpark**:

  * Spark UDF for applying YOLO to large image datasets.
  * Accuracy comparisons across three YOLO model variants.
  * Optimizations using repartitioning, caching, and broadcast variables.

## 📁 Project Structure

```
Vehicle_Detection_Big_Data_Processing/
│
├── data228_code_training_and_kafka/
│   ├── kafka_app/                # Kafka producer and consumers (Telegram + PostgreSQL)
│   └── train_yolo_models/        # Training scripts and config (YOLOv8 + MIO-TCD)
│
├── models/
│   ├── result_first_train/       # COCO-trained model checkpoints (.pt files)
│   └── result_yolov8n/           # YOLOv8n baseline checkpoints
│
├── spark_inference_pipeline/     # PySpark inference and evaluation code (optional folder name)
└── README.md                     # You’re here!
```

## 📊 Results

* **YOLOv8 Final Model Accuracy (MIO-TCD):**

  * Precision: 0.683
  * Recall: 0.549
  * mAP\@0.5: 0.604
  * Final Loss: 0.83 (train), 0.85 (val)

* **Real-Time Alerts:**

  * Telegram notifications triggered after 30s of consistent occupancy.
  * Visual display with OpenCV and logs stored to PostgreSQL for Superset.

* **Spark Accuracy Evaluation:**

  * Final trained YOLOv8s model (on MIO-TCD) outperformed YOLOv8n baseline.
  * Batch inference scaled using 64+ partitions and Spark’s lazy evaluation.

## 📦 Tech Stack

* Ultralytics YOLOv8 (Object Detection)
* Apache Kafka (Real-Time Messaging)
* OpenCV (Video Processing & Visualization)
* Telegram Bot API (Notifications)
* PostgreSQL (Storage)
* PySpark (Batch Inference)

## 🔗 Resources

* [COCO Dataset](https://cocodataset.org/#home)
* [MIO-TCD Dataset](https://tcd.miovision.com/challenge/dataset.html)
* 📂 GitHub Repo: [Vehicle\_Detection\_Big\_Data\_Processing](https://github.com/zincoi172/Vehicle_Detection_Big_Data_Processing)

