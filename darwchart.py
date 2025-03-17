import pandas as pd
import matplotlib
# Cấu hình matplotlib để sử dụng backend không cần Tcl/Tk
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Đường dẫn đúng với các forward slashes hoặc escaped backslashes
try:
    # Kiểm tra các file Spark đã tạo - Spark lưu thành nhiều phần
    import glob
    import os
    
    lr_path = "D:/Big_data/predictions_lr.csv"
    rf_path = "D:/Big_data/predictions_rf.csv"
    cv_path = "D:/Big_data/predictions_cv.csv"
    
    # Tìm tất cả các file CSV trong thư mục
    lr_files = glob.glob(os.path.join(lr_path, "*.csv"))
    rf_files = glob.glob(os.path.join(rf_path, "*.csv"))
    cv_files = glob.glob(os.path.join(cv_path, "*.csv"))
    
    # Đọc và ghép các file
    lr_predictions = None
    rf_predictions = None
    cv_predictions = None
    
    # Kiểm tra có file hay không trước khi đọc
    if lr_files:
        lr_dfs = [pd.read_csv(file) for file in lr_files if os.path.getsize(file) > 0]
        if lr_dfs:
            lr_predictions = pd.concat(lr_dfs, ignore_index=True)
    
    if rf_files:
        rf_dfs = [pd.read_csv(file) for file in rf_files if os.path.getsize(file) > 0]
        if rf_dfs:
            rf_predictions = pd.concat(rf_dfs, ignore_index=True)
    
    if cv_files:
        cv_dfs = [pd.read_csv(file) for file in cv_files if os.path.getsize(file) > 0]
        if cv_dfs:
            cv_predictions = pd.concat(cv_dfs, ignore_index=True)
    
    # Nếu không tìm thấy các file con, thử đọc trực tiếp (nếu Spark đã gộp thành một file)
    if lr_predictions is None and os.path.exists(lr_path):
        try:
            lr_predictions = pd.read_csv(lr_path)
        except:
            pass
    
    if rf_predictions is None and os.path.exists(rf_path):
        try:
            rf_predictions = pd.read_csv(rf_path)
        except:
            pass
            
    if cv_predictions is None and os.path.exists(cv_path):
        try:
            cv_predictions = pd.read_csv(cv_path)
        except:
            pass
    
    # Thử cách khác - đọc trực tiếp các file phần 
    if lr_predictions is None:
        direct_lr = "D:/Big_data/predictions_lr.csv/part-00000-*.csv"
        direct_lr_files = glob.glob(direct_lr)
        if direct_lr_files:
            lr_predictions = pd.read_csv(direct_lr_files[0])
    
    if rf_predictions is None:
        direct_rf = "D:/Big_data/predictions_rf.csv/part-00000-*.csv"
        direct_rf_files = glob.glob(direct_rf)
        if direct_rf_files:
            rf_predictions = pd.read_csv(direct_rf_files[0])
    
    if cv_predictions is None:
        direct_cv = "D:/Big_data/predictions_cv.csv/part-00000-*.csv"
        direct_cv_files = glob.glob(direct_cv)
        if direct_cv_files:
            cv_predictions = pd.read_csv(direct_cv_files[0])
    
    # Kiểm tra xem đã đọc được dữ liệu chưa
    if lr_predictions is None or rf_predictions is None or cv_predictions is None:
        print("❌ Không thể đọc một hoặc nhiều file dự đoán. Hãy kiểm tra đường dẫn và cấu trúc thư mục.")
        
        # Hiển thị thông tin về các file tìm thấy
        print("\nThông tin debug:")
        print(f"lr_files: {lr_files}")
        print(f"rf_files: {rf_files}")
        print(f"cv_files: {cv_files}")
        
        # Tìm kiếm tất cả các file có thể liên quan trong thư mục
        all_csv = glob.glob("D:/Big_data/**/*.csv", recursive=True)
        print(f"\nTất cả các file CSV tìm thấy trong D:/Big_data/:")
        for csv_file in all_csv:
            print(f"  - {csv_file}")
            
        # Kết thúc
        import sys
        sys.exit(1)
        
    # Đặt tên cho mỗi DataFrame để dễ phân biệt
    lr_predictions['model'] = 'Linear Regression'
    rf_predictions['model'] = 'Random Forest'
    cv_predictions['model'] = 'Cross Validation'
    
    # Kết hợp dữ liệu từ tất cả các mô hình
    all_predictions = pd.concat([lr_predictions, rf_predictions, cv_predictions])
    
    # Thiết lập style cho biểu đồ
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # 1. So sánh giá trị thực tế với dự đoán của từng mô hình
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Dữ liệu cho mỗi mô hình
    models_data = {
        'Linear Regression': lr_predictions,
        'Random Forest': rf_predictions,
        'Cross Validation': cv_predictions
    }
    
    # Vẽ biểu đồ scatter cho từng mô hình
    for i, (model_name, data) in enumerate(models_data.items()):
        # Lọc dữ liệu để không hiển thị quá nhiều điểm
        sample = data.sample(n=min(500, len(data)))
        
        # Tính R2 và RMSE
        r2 = r2_score(sample['Confirmed'], sample['prediction'])
        rmse = np.sqrt(mean_squared_error(sample['Confirmed'], sample['prediction']))
        
        # Vẽ biểu đồ
        axes[i].scatter(sample['Confirmed'], sample['prediction'], alpha=0.5)
        
        # Vẽ đường xu hướng (đường thẳng lý tưởng)
        max_val = max(sample['Confirmed'].max(), sample['prediction'].max())
        axes[i].plot([0, max_val], [0, max_val], 'r--')
        
        axes[i].set_title(f'{model_name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        axes[i].set_xlabel('Giá trị thực tế')
        axes[i].set_ylabel('Giá trị dự đoán')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('D:/Big_data/actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # 2. Biểu đồ phân phối sai số cho từng mô hình
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (model_name, data) in enumerate(models_data.items()):
        # Tính sai số
        data['error'] = data['prediction'] - data['Confirmed']
        
        # Vẽ biểu đồ phân phối sai số (histogram thay vì seaborn)
        axes[i].hist(data['error'], bins=30, alpha=0.7, density=True)
        
        # Tính sai số trung bình
        mean_error = data['error'].mean()
        median_error = data['error'].median()
        
        axes[i].axvline(mean_error, color='r', linestyle='--', label=f'Trung bình: {mean_error:.2f}')
        axes[i].axvline(median_error, color='g', linestyle='--', label=f'Trung vị: {median_error:.2f}')
        
        axes[i].set_title(f'Phân phối sai số - {model_name}')
        axes[i].set_xlabel('Sai số (Dự đoán - Thực tế)')
        axes[i].set_ylabel('Mật độ')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('D:/Big_data/error_distribution.png', dpi=300)
    plt.close()
    
    # 3. So sánh hiệu suất các mô hình
    # Tạo DataFrame để lưu trữ các metric đánh giá
    metrics = []
    
    for model_name, data in models_data.items():
        r2 = r2_score(data['Confirmed'], data['prediction'])
        rmse = np.sqrt(mean_squared_error(data['Confirmed'], data['prediction']))
        
        metrics.append({
            'Model': model_name,
            'R2': r2,
            'RMSE': rmse
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Vẽ biểu đồ cột so sánh R2
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df['Model']))
    width = 0.35
    
    # Trục chính cho R2
    bar1 = ax1.bar(x - width/2, metrics_df['R2'], width, label='R²', color='skyblue')
    ax1.set_ylabel('R² (càng cao càng tốt)', color='blue')
    ax1.set_ylim(0, 1.05)
    
    # Trục thứ hai cho RMSE
    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + width/2, metrics_df['RMSE'], width, label='RMSE', color='lightcoral')
    ax2.set_ylabel('RMSE (càng thấp càng tốt)', color='red')
    
    # Thêm nhãn và tiêu đề
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'])
    ax1.set_title('So sánh hiệu suất các mô hình')
    
    # Thêm giá trị lên đầu cột
    def add_labels(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bar1, ax1)
    add_labels(bar2, ax2)
    
    # Thêm chú thích
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('D:/Big_data/model_comparison.png', dpi=300)
    plt.close()
    
    # Tạo thêm một báo cáo về các chỉ số hiệu suất
    with open('D:/Big_data/model_performance_report.txt', 'w', encoding="utf-8") as f:
        f.write("BÁO CÁO HIỆU SUẤT MÔ HÌNH\n")
        f.write("=" * 50 + "\n\n")
        
        for index, row in metrics_df.iterrows():
            f.write(f"Mô hình: {row['Model']}\n")
            f.write(f"R² Score: {row['R2']:.6f}\n")
            f.write(f"RMSE: {row['RMSE']:.6f}\n")
            f.write("-" * 50 + "\n\n")
        
        # Xác định mô hình tốt nhất
        best_r2_model = metrics_df.loc[metrics_df['R2'].idxmax(), 'Model']
        best_rmse_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
        
        f.write(f"Mô hình có R² cao nhất: {best_r2_model}\n")
        f.write(f"Mô hình có RMSE thấp nhất: {best_rmse_model}\n")
        
    print("✅ Đã tạo các biểu đồ và lưu vào thư mục D:/Big_data/")
    print("📊 Các file biểu đồ:")
    print("  - actual_vs_predicted.png")
    print("  - error_distribution.png")
    print("  - model_comparison.png")
    print("  - model_performance_report.txt")

except Exception as e:
    print(f"❌ Đã xảy ra lỗi: {e}")
    import traceback
    traceback.print_exc()
    print("\nHãy đảm bảo:")
    print("1. Đã chạy mã nguồn dự đoán và tạo các file CSV")
    print("2. Đường dẫn đến các file CSV chính xác")
    print("3. Tất cả các thư viện cần thiết đã được cài đặt (pandas, matplotlib, numpy, sklearn)")