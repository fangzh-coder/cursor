import pandas as pd

# --- 配置您的文件路径 ---
# 输入的 Parquet 文件路径
parquet_file_path = 'D:/Investment/20250628 quant/hist_data/000001.SH_1min.parquet'
# 输出的 Excel 文件路径
excel_file_path = 'D:/Investment/20250628 quant/hist_data/000001.SH_1min.xlsx'

# 读取 Parquet 文件到 pandas DataFrame
print(f"正在读取 Parquet 文件: {parquet_file_path}...")
df = pd.read_parquet(parquet_file_path)
print("文件读取成功！")

# 将 DataFrame 保存为 Excel 文件
# index=False 表示不将 DataFrame 的索引写入到 Excel 文件中
# engine='openpyxl' 是指定使用 openpyxl 引擎来写入 .xlsx 文件
print(f"正在将数据转换为 Excel 文件: {excel_file_path}...")
df.to_excel(excel_file_path, index=False, engine='openpyxl')

print("转换完成！Excel 文件已成功保存。")