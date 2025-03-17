from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, max, avg, expr, round
import matplotlib.pyplot as plt
import pandas as pd

# Tạo Spark Session
spark = SparkSession.builder.appName("COVID-19 Analysis with Spark UI")\
    .config("spark.driver.host", "127.0.0.1")\
    .config("spark.driver.bindAddress", "127.0.0.1")\
    .getOrCreate()

# Đọc dữ liệu
file_path = "country_wise_latest.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Chuyển đổi kiểu dữ liệu
df = df.withColumn("Confirmed", col("Confirmed").cast("int"))\
       .withColumn("Deaths", col("Deaths").cast("int"))\
       .withColumn("Recovered", col("Recovered").cast("int"))\
       .withColumn("Confirmed last week", col("Confirmed last week").cast("int"))\
       .withColumn("1 week change", col("1 week change").cast("int"))\
       .withColumn("1 week % increase", col("1 week % increase").cast("float"))

# Tính toán các chỉ số mới
df = df.withColumn("Fatality Rate (%)", expr("CASE WHEN Confirmed > 0 THEN (Deaths / Confirmed) * 100 ELSE 0 END"))
df = df.withColumn("Recovery Rate (%)", expr("CASE WHEN Confirmed > 0 THEN (Recovered / Confirmed) * 100 ELSE 0 END"))

# 1️⃣ Tổng số ca nhiễm, tử vong, hồi phục theo WHO Region
region_summary = df.groupBy("WHO Region").agg(
    sum("Confirmed").alias("Total Confirmed"),
    sum("Deaths").alias("Total Deaths"),
    sum("Recovered").alias("Total Recovered"),
    round(avg("Fatality Rate (%)"), 2).alias("Avg Fatality Rate (%)")
)

# 2️⃣ Quốc gia có số ca nhiễm, tử vong, hồi phục cao nhất
worst_country = df.orderBy(col("Confirmed").desc()).select("Country/Region", "Confirmed").first()
deadliest_country = df.orderBy(col("Deaths").desc()).select("Country/Region", "Deaths").first()
highest_recovery_country = df.orderBy(col("Recovered").desc()).select("Country/Region", "Recovered").first()

# 3️⃣ Quốc gia có tỷ lệ tử vong cao nhất (%)
highest_fatality_rate_country = df.orderBy(col("Fatality Rate (%)").desc()).select("Country/Region", "Fatality Rate (%)").first()

# 🔹 Phân Tích Xu Hướng COVID-19
# 1️⃣ Xu hướng gia tăng ca nhiễm theo tuần
total_new_cases = df.selectExpr("sum(`1 week change`) as Total_New_Cases").collect()[0]["Total_New_Cases"]
highest_weekly_country = df.orderBy(col("1 week change").desc()).select("Country/Region", "1 week change").first()
avg_weekly_increase = df.selectExpr("round(avg(`1 week % increase`),2) as Avg_Weekly_Increase").collect()[0]["Avg_Weekly_Increase"]
highest_regions = df.groupBy("WHO Region").agg(sum("1 week change").alias("Total_New_Cases")).orderBy(col("Total_New_Cases").desc()).limit(3).toPandas()

# 2️⃣ Xu hướng tử vong và hồi phục
avg_death_rate = df.selectExpr("round(avg(Deaths / 100),2) as Avg_Death_Rate").collect()[0]["Avg_Death_Rate"]
avg_recovery_rate = df.selectExpr("round(avg(Recovered / 100),2) as Avg_Recovery_Rate").collect()[0]["Avg_Recovery_Rate"]
highest_death_country = df.orderBy(col("Fatality Rate (%)").desc()).select("Country/Region", "Fatality Rate (%)").first()
highest_recovery_country = df.orderBy(col("Recovery Rate (%)").desc()).select("Country/Region", "Recovery Rate (%)").first()

# 🔹 Ghi báo cáo vào file
with open("covid_analysis_report.txt", "w", encoding="utf-8") as file:
    file.write("📊 COVID-19 Data Analysis Report\n")
    file.write("=" * 60 + "\n")
    file.write(f"🟢 Quốc gia có số ca nhiễm cao nhất: {worst_country['Country/Region']} ({worst_country['Confirmed']} cases)\n")
    file.write(f"🔴 Quốc gia có số ca tử vong cao nhất: {deadliest_country['Country/Region']} ({deadliest_country['Deaths']} deaths)\n")
    file.write(f"⚠ Quốc gia có tỷ lệ tử vong cao nhất: {highest_fatality_rate_country['Country/Region']} ({highest_fatality_rate_country['Fatality Rate (%)']:.2f}%)\n")
    file.write("=" * 60 + "\n")
    file.write(f"📌 Xu hướng gia tăng ca nhiễm theo tuần: {total_new_cases} ca mới\n")
    file.write(f"📌 Quốc gia có mức tăng cao nhất: {highest_weekly_country['Country/Region']} ({highest_weekly_country['1 week change']} ca mới)\n")
    file.write(f"📌 Mức tăng trung bình: {avg_weekly_increase}%\n")
    file.write(f"📌 Các khu vực có mức tăng cao nhất:\n{highest_regions.to_string()}\n")
    file.write("=" * 60 + "\n")
    file.write(f"📌 Tỷ lệ tử vong trung bình trên 100 ca: {avg_death_rate}%\n")
    file.write(f"📌 Tỷ lệ hồi phục trung bình trên 100 ca: {avg_recovery_rate}%\n")
    file.write(f"📌 Quốc gia có tỷ lệ tử vong cao nhất: {highest_death_country['Country/Region']} ({highest_death_country['Fatality Rate (%)']:.2f}%)\n")
    file.write(f"📌 Quốc gia có tỷ lệ hồi phục cao nhất: {highest_recovery_country['Country/Region']} ({highest_recovery_country['Recovery Rate (%)']:.2f}%)\n")
    file.write("=" * 60 + "\n")

# Hiển thị Spark UI
print("Spark UI available at:", spark.sparkContext.uiWebUrl)

# Dừng Spark
spark.stop()