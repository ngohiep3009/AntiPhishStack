import pandas as pd

phish = pd.read_csv("data/raw/verified_online.csv")

phish_urls = phish[['url']].copy()
phish_urls['label'] = 1

clean = pd.read_csv("data/raw/top-1m.csv", names=["rank", "domain"])

clean_urls = clean[['domain']].copy()
clean_urls['url'] = "http://" + clean_urls['domain']
clean_urls = clean_urls[['url']]
clean_urls['label'] = 0

# lấy mỗi bên 5000 mẫu
phish_urls = phish_urls.head(5000)
clean_urls = clean_urls.head(5000)

# ghép 2 loại dữ liệu
dataset = pd.concat([phish_urls, clean_urls], ignore_index=True)

# xóa dòng rỗng
dataset = dataset.dropna(subset=["url"])

# xóa URL trùng
dataset = dataset.drop_duplicates(subset=["url"])

# trộn ngẫu nhiên dữ liệu
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# lưu file
dataset.to_csv("data/processed/dataset.csv", index=False)

# kiểm tra
print(dataset.head())
print("Total samples:", len(dataset))
print(dataset["label"].value_counts())