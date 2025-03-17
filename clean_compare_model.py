from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator


spark = SparkSession.builder \
    .appName("COVID Prediction") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

print("Spark UI đang chạy tại:", spark.sparkContext.uiWebUrl)


data_path = "D:/Big_data/usa_county_wise.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)


df_cleaned = df.dropna(subset=["Admin2", "Province_State", "Country_Region", "Confirmed", "Deaths"])
df_cleaned = df_cleaned.filter(col("Confirmed") >= 0)

# 4️⃣ Tạo vector đặc trưng cho mô hình học máy
assembler = VectorAssembler(inputCols=["Confirmed", "Deaths"], outputCol="features")
df_transformed = assembler.transform(df_cleaned)

# 5️⃣ Chia dữ liệu thành tập huấn luyện và kiểm tra
train_data, test_data = df_transformed.randomSplit([0.8, 0.2], seed=1234)

# 6️⃣ Thử mô hình Linear Regression với Regularization
lr = LinearRegression(featuresCol="features", labelCol="Confirmed", regParam=0.1)

# Huấn luyện mô hình Linear Regression với Regularization
lr_model = lr.fit(train_data)

# Đánh giá mô hình Linear Regression trên tập kiểm tra
test_results_lr = lr_model.evaluate(test_data)
print("Linear Regression - R2:", test_results_lr.r2)
print("Linear Regression - RMSE:", test_results_lr.rootMeanSquaredError)

# 7️⃣ Thử mô hình Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol="Confirmed")

# Huấn luyện mô hình Random Forest
rf_model = rf.fit(train_data)

# Dự đoán trên tập kiểm tra
predictions_rf = rf_model.transform(test_data)

# Đánh giá mô hình Random Forest trên tập kiểm tra bằng RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="Confirmed", predictionCol="prediction", metricName="r2")
r2_rf = evaluator.evaluate(predictions_rf)
print("Random Forest - R2:", r2_rf)

# Đánh giá RMSE
evaluator.setMetricName("rmse")
rmse_rf = evaluator.evaluate(predictions_rf)
print("Random Forest - RMSE:", rmse_rf)

# 8️⃣ Sử dụng Cross-validation để tối ưu tham số cho mô hình Linear Regression
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
    .addGrid(lr.elasticNetParam, [0.8, 0.5, 0.2]) \
    .build()

# Khởi tạo CrossValidator
evaluator = RegressionEvaluator(labelCol="Confirmed", metricName="rmse")
crossval = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

# Huấn luyện mô hình với Cross-validation
cv_model = crossval.fit(train_data)

# Lấy mô hình tốt nhất từ cross-validation
best_lr_model = cv_model.bestModel

# Đánh giá mô hình tốt nhất trên tập kiểm tra
predictions_cv = cv_model.transform(test_data)
evaluator.setMetricName("r2")
r2_cv = evaluator.evaluate(predictions_cv)
evaluator.setMetricName("rmse")
rmse_cv = evaluator.evaluate(predictions_cv)
print("Cross-validation Best Model - R2:", r2_cv)
print("Cross-validation Best Model - RMSE:", rmse_cv)

# 9️⃣ Dự đoán trên tập kiểm tra với tất cả các mô hình
predictions_lr = lr_model.transform(test_data)
predictions_rf = rf_model.transform(test_data)
predictions_cv = cv_model.transform(test_data)

# 10️⃣ Lưu kết quả dự đoán
predictions_lr.select("Confirmed", "prediction").write.csv("D:/Big_data/predictions_lr.csv", header=True, mode="overwrite")
predictions_rf.select("Confirmed", "prediction").write.csv("D:/Big_data/predictions_rf.csv", header=True, mode="overwrite")
predictions_cv.select("Confirmed", "prediction").write.csv("D:/Big_data/predictions_cv.csv", header=True, mode="overwrite")

print("✅ Kết quả dự đoán đã được lưu tại D:/Big_data/predictions_lr.csv, predictions_rf.csv, predictions_cv.csv")

# 11️⃣ Caching dữ liệu để tối ưu hóa quá trình
df_transformed.cache()

# 12️⃣ Giữ chương trình chạy để kiểm tra Spark UI
input("Nhấn Enter để thoát...")

# 13️⃣ Dừng SparkSession khi kết thúc
spark.stop() 