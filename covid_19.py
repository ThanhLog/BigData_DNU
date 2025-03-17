from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, abs
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

# 🔥 Tạo SparkSession với Spark UI (cổng 4040)
spark = SparkSession.builder \
    .appName("COVID-19 Prediction with Spark UI") \
    .config("spark.ui.port", "4041") \
    .getOrCreate()

# 🚀 Giảm mức log để dễ quan sát trên Spark UI
spark.sparkContext.setLogLevel("ERROR")

# ⏳ Theo dõi thời gian thực thi
start_time = time.time()

print("📥 Đọc dữ liệu...")
df = spark.read.csv(r"D:\Big_data\usa_county_wise_cleaned.csv", header=True, inferSchema=True)

# 🛠 Chuyển đổi kiểu dữ liệu
df = df.withColumn("Confirmed", col("Confirmed").cast(DoubleType())) \
       .withColumn("Deaths", col("Deaths").cast(DoubleType())) \
       .withColumn("Lat", col("Lat").cast(DoubleType())) \
       .withColumn("Long_", col("Long_").cast(DoubleType()))

# 🧹 Xử lý dữ liệu bị thiếu
df = df.dropna(subset=["Confirmed", "Deaths", "Lat", "Long_"])

# 🎯 Chuẩn bị đặc trưng nhóm theo bang
def prepare_features(dataframe):
    print("🔄 Trích xuất đặc trưng...")
    grouped_df = dataframe.groupBy("Province_State").agg(
        avg("Confirmed").alias("Avg_Confirmed"),
        sum("Confirmed").alias("Total_Confirmed"),
        avg("Deaths").alias("Avg_Deaths"),
        sum("Deaths").alias("Total_Deaths"),
        avg("Lat").alias("Avg_Latitude"),
        avg("Long_").alias("Avg_Longitude")
    )
    return grouped_df

# 🔄 Chuẩn bị dữ liệu huấn luyện
def prepare_model_data(dataframe):
    feature_cols = ["Avg_Latitude", "Avg_Longitude", "Avg_Confirmed", "Total_Confirmed", "Avg_Deaths", "Total_Deaths"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

    dataframe = dataframe.withColumn("label", col("Total_Confirmed"))
    return dataframe, assembler, scaler

# 🌳 Xây dựng mô hình dự đoán
def build_prediction_model(assembler, scaler):
    print("📡 Xây dựng mô hình Random Forest...")
    rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="label", numTrees=50, maxDepth=10)
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    return pipeline

# 🎯 Đánh giá mô hình
def evaluate_model(predictions):
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print(f"✅ RMSE: {rmse}")
    print(f"📊 R² Score: {r2}")

# 🔍 Hiển thị kết quả dự đoán
def analyze_predictions(predictions):
    print("\n📊 Một số kết quả dự đoán:")
    predictions.select("Province_State", "label", "prediction").show(10)

    print("\n🚨 Sai số dự đoán lớn nhất:")
    predictions.withColumn("Prediction_Error", abs(col("label") - col("prediction"))) \
               .orderBy(col("Prediction_Error").desc()) \
               .select("Province_State", "label", "prediction", "Prediction_Error") \
               .show(10)

# 🏁 Chạy quá trình dự đoán
def main():
    prepared_df = prepare_features(df)

    print("📊 Chia tập dữ liệu thành Train/Test...")
    train_data, test_data = prepared_df.randomSplit([0.8, 0.2], seed=42)

    model_train, assembler, scaler = prepare_model_data(train_data)
    model_test, _, _ = prepare_model_data(test_data)

    pipeline = build_prediction_model(assembler, scaler)

    print("🚀 Bắt đầu huấn luyện mô hình...")
    model = pipeline.fit(model_train)

    print("📡 Thực hiện dự đoán trên tập kiểm tra...")
    predictions = model.transform(model_test)

    print("📈 Đánh giá mô hình...")
    evaluate_model(predictions)

    print("🔍 Phân tích kết quả dự đoán...")
    analyze_predictions(predictions)

    # ⏱ Kết thúc thời gian thực thi
    end_time = time.time()
    print(f"⏳ Thời gian thực thi: {round(end_time - start_time, 2)} giây")
    print(spark.sparkContext.uiWebUrl)
main()

# 🛑 Đóng SparkSession
spark.stop()
