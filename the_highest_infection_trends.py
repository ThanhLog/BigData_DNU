from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import numpy as np
import time

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("COVID-19 State/County Prediction") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

# Giảm log để dễ đọc hơn
spark.sparkContext.setLogLevel("ERROR")

# Đường dẫn dữ liệu
data_path = r"D:\Big_data\usa_county_wise_cleaned.csv"

# Bắt đầu đo thời gian
start_time = time.time()

# Đọc dữ liệu
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Làm sạch và chuẩn bị dữ liệu
def preprocess_data(dataframe):
    # Loại bỏ dòng có giá trị null
    df_cleaned = dataframe.dropna(subset=[
        "Province_State", "Admin2", "Lat", "Long_", 
        "Confirmed", "Deaths"
    ])
    
    # Chuyển đổi kiểu dữ liệu số
    df_cleaned = df_cleaned.withColumn("Lat", col("Lat").cast("double")) \
                           .withColumn("Long_", col("Long_").cast("double")) \
                           .withColumn("Confirmed", col("Confirmed").cast("double")) \
                           .withColumn("Deaths", col("Deaths").cast("double"))
    
    return df_cleaned

# Xử lý dữ liệu
df_processed = preprocess_data(df)

# Chuẩn bị đặc trưng cho mô hình
def prepare_features(dataframe):
    # Mã hóa các đặc trưng phân loại
    state_indexer = StringIndexer(inputCol="Province_State", outputCol="State_Index", handleInvalid="keep")
    county_indexer = StringIndexer(inputCol="Admin2", outputCol="County_Index", handleInvalid="keep")
    
    # Chọn các đặc trưng
    feature_columns = ["Lat", "Long_", "State_Index", "County_Index", "Deaths"]
    
    # Tập hợp các đặc trưng
    assembler = VectorAssembler(
        inputCols=feature_columns, 
        outputCol="features", 
        handleInvalid="skip"
    )
    
    # Chuẩn hóa đặc trưng
    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaled_features", 
        withStd=True, 
        withMean=True
    )
    
    # Hồi quy Gradient Boosting
    regressor = GBTRegressor(
        featuresCol="scaled_features", 
        labelCol="Confirmed", 
        maxDepth=5, 
        maxBins=32
    )
    
    # Tạo đường ống ML
    pipeline = Pipeline(stages=[
        state_indexer,
        county_indexer,
        assembler, 
        scaler, 
        regressor
    ])
    
    return pipeline

# Chia dữ liệu train/test
(train_data, test_data) = df_processed.randomSplit([0.8, 0.2], seed=42)

# Xây dựng và huấn luyện mô hình
pipeline = prepare_features(df_processed)
model = pipeline.fit(train_data)

# Dự đoán và đánh giá
predictions = model.transform(test_data)

# Đánh giá mô hình
evaluator = RegressionEvaluator(
    labelCol="Confirmed", 
    predictionCol="prediction", 
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Dự đoán mẫu
print("\nDự đoán mẫu:")
sample_predictions = predictions.select(
    "Province_State", 
    "Admin2", 
    "Confirmed", 
    "prediction"
).show(10)

# Tìm các bang/quận có xu hướng nhiễm cao nhất
print("\nCác khu vực có xu hướng nhiễm cao nhất:")
top_regions = predictions.groupBy("Province_State", "Admin2") \
    .agg(mean("prediction").alias("avg_predicted_confirmed")) \
    .orderBy(col("avg_predicted_confirmed").desc()) \
    .show(10)

# Kết thúc và tính thời gian chạy
end_time = time.time()
print(f"\nThời gian chạy: {end_time - start_time:.2f} giây")

# Đóng SparkSession
spark.stop()