from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import os
import random
import time
import cv2
from ultralytics import YOLO
from PIL import Image

# Start timer
start_time = time.time()

# Create Spark session (64 partitions for parallelism)
spark = SparkSession.builder \
    .appName("YOLOInference") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "64") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Important vehicle classes
important_classes = [
    'articulated_truck', 'pickup_truck', 'single_unit_truck',
    'car', 'bus'
]

folder_rules = {
    'articulated_truck': ['Articulated truck'],
    'pickup_truck': ['Pickup truck', 'Car'],
    'single_unit_truck': ['Single unit truck', 'Car'],
    'car': ['Car'],
    'bus': ['Bus']
}
# Folder rules
# folder_rules = {
#     'articulated_truck': ['car', 'truck'],
#     'pickup_truck': ['car', 'truck'],
#     'single_unit_truck': ['car', 'truck'],
#     'car': ['car'],
#     'bus': ['bus']
# }

# Base folder
base_folder = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/train'
right_predictions_folder = os.path.join(base_folder, 'right_predictions_final_best_pt')
wrong_predictions_folder = os.path.join(base_folder, 'wrong_predictions_final_best_pt')
os.makedirs(right_predictions_folder, exist_ok=True)
os.makedirs(wrong_predictions_folder, exist_ok=True)

# Load up to 50 images per folder
image_paths = []
for folder in important_classes:
    folder_path = os.path.join(base_folder, folder)
    if os.path.exists(folder_path):
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        selected = random.sample(files, min(100, len(files)))
        image_paths.extend(selected)

print(f"✅ Total images collected: {len(image_paths)}")
if len(image_paths) == 0:
    print("⚠️ No images found. Exiting.")
    exit(1)

# Create Spark DataFrame
image_df = spark.createDataFrame([(path,) for path in image_paths], ['image_path']).repartition(64).cache()

# Broadcast model path
model_path = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/final_best.pt'
broadcast_model_path = spark.sparkContext.broadcast(model_path)

# === GLOBAL model ===
global_model = None

# Define inference UDF with image saving for right/wrong predictions
def run_inference(image_path):
    global global_model
    from ultralytics import YOLO
    from PIL import Image
    import os
    import cv2

    model_file = broadcast_model_path.value

    if global_model is None:
        try:
            if not os.path.exists(model_file):
                print(f"⚠️ Model not found: {model_file}")
                return None
            global_model = YOLO(model_file)
            print(f"✅ Model loaded in executor.")
        except Exception as model_error:
            print(f"⚠️ Model loading error: {model_error}")
            return None

    folder_name = os.path.basename(os.path.dirname(image_path))

    if folder_name not in folder_rules:
        return folder_name, 0, 0

    try:
        if not os.path.exists(image_path):
            print(f"⚠️ Missing image: {image_path}")
            return folder_name, 0, 1
        img = Image.open(image_path)
        img.verify()
    except Exception as img_error:
        print(f"⚠️ Bad image skipped: {image_path}, Error: {img_error}")
        return folder_name, 0, 1

    try:
        results = global_model(image_path, conf=0.1)
        result = results[0]
        names = global_model.names
        detected_labels = [names[int(cls)] for cls in result.boxes.cls]

        allowed_labels = folder_rules[folder_name]
        img_file = os.path.basename(image_path)
        img_cv = cv2.imread(image_path)

        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls)]
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if any(label in allowed_labels for label in detected_labels):
            save_folder = os.path.join(right_predictions_folder, folder_name)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, img_file)
            cv2.imwrite(save_path, img_cv)
            return folder_name, 1, 0
        else:
            save_folder = os.path.join(wrong_predictions_folder, folder_name)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, img_file)
            cv2.imwrite(save_path, img_cv)
            return folder_name, 0, 1

    except Exception as infer_error:
        print(f"⚠️ Inference error: {image_path}, Error: {infer_error}")
        return folder_name, 0, 1

# Define output schema
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

# Group by folder
summary_df = final_df.groupBy("folder") \
    .sum("correct", "incorrect") \
    .withColumnRenamed("sum(correct)", "Correct") \
    .withColumnRenamed("sum(incorrect)", "Incorrect") \
    .withColumn(
        "Accuracy (%)",
        spark_round((col("Correct") / (col("Correct") + col("Incorrect"))) * 100, 2)
    )

# Show output
summary_df.show(truncate=False)

# Save results
output_path = '/Users/daivo/Desktop/SJSU 2024/Spring 2025/DATA 228/228 Project/prediction_delete3'
summary_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

print(f"✅ Results saved to {output_path}")
print(f"✅ Finished in {round((time.time() - start_time)/60, 2)} minutes.")
