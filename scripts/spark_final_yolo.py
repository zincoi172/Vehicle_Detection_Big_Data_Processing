from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os
import random
import time
from ultralytics import YOLO
from PIL import Image

# Start timer
start_time = time.time()

# Create Spark session
spark = SparkSession.builder \
    .appName("YOLOInference") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "64") \
    .config("spark.executor.memory", "2g") \
    .config("spark.speculation", "true") \
    .getOrCreate()

# Define vehicle classes or folders to evaluate
important_classes = ['articulated_truck', 'pickup_truck', 'single_unit_truck', 'car', 'bus']

# 🔁 Updated folder_rules to match model class names from `final_best.pt`
folder_rules = {
    'articulated_truck': ['Articulated truck'],
    'pickup_truck': ['Pickup truck', 'Car'],
    'single_unit_truck': ['Single unit truck', 'Car'],
    'car': ['Car'],
    'bus': ['Bus']
}

# This folder rules use for first_train.pt and yolov8n.pt
# folder_rules = {
#     'articulated_truck': ['car', 'truck'],
#     'pickup_truck': ['car', 'truck'],
#     'single_unit_truck': ['car', 'truck'],
#     'car': ['car'],
#     'bus': ['bus']
# }

# Load up to 50 images per folder
base_folder = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/train'
image_paths = []

for folder in important_classes:
    folder_path = os.path.join(base_folder, folder)
    if os.path.exists(folder_path):
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(files)

print(f"✅ Total images collected: {len(image_paths)}")
if len(image_paths) == 0:
    print("⚠️ No images found. Exiting.")
    exit(1)

# Smart repartitioning
num_partitions = max(64, len(image_paths) // 500)
image_df = spark.createDataFrame([(path,) for path in image_paths], ['image_path']) \
    .repartition(num_partitions).cache()

# Broadcast model path
model_path = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/final_best.pt' # Or first_train.pt or yolov8n.pt
broadcast_model_path = spark.sparkContext.broadcast(model_path)

# === GLOBAL model ===
global_model = None

# UDF to run inference and save misclassified images with bounding boxes
def run_inference(image_path):
    global global_model
    import os
    import cv2
    from ultralytics import YOLO
    from PIL import Image

    model_file = broadcast_model_path.value

    if global_model is None:
        try:
            if not os.path.exists(model_file):
                print(f"⚠️ Model not found: {model_file}")
                return None
            global_model = YOLO(model_file)
            print(f"✅ Model loaded in executor")
        except Exception as model_error:
            print(f"⚠️ Model load error: {model_error}")
            return None

    folder_name = os.path.basename(os.path.dirname(image_path))
    if folder_name not in folder_rules:
        return folder_name, 0, 0

    try:
        if not os.path.exists(image_path):
            return folder_name, 0, 1
        img = Image.open(image_path)
        img.verify()
    except Exception:
        return folder_name, 0, 1

    try:
        results = global_model(image_path, conf=0.1)
        result = results[0]
        names = global_model.names
        detected_labels = [names[int(cls)] for cls in result.boxes.cls]

        allowed_labels = folder_rules[folder_name]
        if any(label in allowed_labels for label in detected_labels):
            return folder_name, 1, 0
        else:
            save_folder = os.path.join(base_folder, "wrong_predictions_final_best_pt", folder_name)
            os.makedirs(save_folder, exist_ok=True)
            img_file = os.path.basename(image_path)
            img = cv2.imread(image_path)

            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = names[int(cls)]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            save_path = os.path.join(save_folder, img_file)
            cv2.imwrite(save_path, img)

            return folder_name, 0, 1
    except Exception:
        return folder_name, 0, 1

# Define return schema
schema = StructType([
    StructField("folder", StringType(), True),
    StructField("correct", IntegerType(), True),
    StructField("incorrect", IntegerType(), True)
])

# Apply UDF
inference_udf = udf(run_inference, schema)
results_df = image_df.withColumn("result", inference_udf("image_path")).cache()

# Flatten results
final_df = results_df.select(
    col("result.folder").alias("folder"),
    col("result.correct").alias("correct"),
    col("result.incorrect").alias("incorrect")
)

# Group and summarize
summary_df = final_df.groupBy("folder") \
    .sum("correct", "incorrect") \
    .withColumnRenamed("sum(correct)", "Correct") \
    .withColumnRenamed("sum(incorrect)", "Incorrect") \
    .withColumn("Accuracy (%)", spark_round((col("Correct") / (col("Correct") + col("Incorrect"))) * 100, 2))

# Save as CSV
output_path = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/prediction_final_best_pt'
summary_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

# Show a preview
summary_df.limit(5).show(truncate=False)

print(f"✅ Results saved to {output_path}")
print(f"✅ Finished in {round((time.time() - start_time)/60, 2)} minutes.")
