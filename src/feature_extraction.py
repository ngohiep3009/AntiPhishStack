import pandas as pd
import re
from urllib.parse import urlparse

df = pd.read_csv("data/processed/dataset_clean.csv")

# dùng url_clean để trích feature
urls = df["url_clean"]

# ===== BASIC FEATURES =====
df["url_length"] = urls.apply(len)
df["num_dots"] = urls.str.count(r"\.")
df["num_dash"] = urls.str.count("-")
df["num_slash"] = urls.str.count("/")
df["num_at"] = urls.str.count("@")

# ===== ADVANCED FEATURES =====

# 1. có IP address không
def has_ip(url):
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(ip_pattern, url) else 0

df["has_ip"] = urls.apply(has_ip)

# 2. có từ nghi ngờ không
def suspicious_words(url):
    keywords = ["login", "verify", "update", "bank", "secure", "account"]
    return 1 if any(word in url for word in keywords) else 0

df["suspicious_word"] = urls.apply(suspicious_words)

# 3. số subdomain
def count_subdomain(url):
    domain = urlparse("http://" + url).netloc
    return max(0, domain.count(".") - 1)

df["subdomain_count"] = urls.apply(count_subdomain)

# 4. có https không (dựa vào url gốc)
df["https"] = df["url"].str.startswith("https").astype(int)

# ===== LƯU FILE =====
df.to_csv("data/processed/dataset_features.csv", index=False)

print(df.head())
print("Total samples:", len(df))