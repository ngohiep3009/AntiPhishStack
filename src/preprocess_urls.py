import pandas as pd

# đọc dataset vừa tạo
df = pd.read_csv("data/processed/dataset.csv")

# hàm làm sạch URL
def clean_url(url):
    url = str(url).strip().lower()
    url = url.replace("http://", "")
    url = url.replace("https://", "")
    url = url.replace("www.", "")
    return url

# tạo cột mới
df["url_clean"] = df["url"].apply(clean_url)

# xóa trùng theo url_clean
df = df.drop_duplicates(subset=["url_clean"])

# xóa dòng lỗi
df = df.dropna(subset=["url_clean"])

# lưu file mới
df.to_csv("data/processed/dataset_clean.csv", index=False)

# kiểm tra
print(df.head())
print("Total samples:", len(df))
print(df["label"].value_counts())