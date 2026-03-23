import pandas as pd
import numpy as np
import re
from tldextract import extract
from sklearn.model_selection import train_test_split

data = {
    'url': [
        'https://www.google.com',
        'http://wellsfargo.com-login-update.verify.com/secure',
        'https://www.facebook.com',
        'http://192.168.1.1/admin',
        'https://www.google.com',  
        'nan',                      
        'http://secure-bank-access.tk/login'
    ],
    'label': [0, 1, 0, 1, 0, 0, 1] 
}
df = pd.DataFrame(data)

print("--- Dữ liệu ban đầu ---")
print(df)

# 2. LÀM SẠCH DỮ LIỆU (Data Cleaning)
# Loại bỏ giá trị trống (NaN)
df = df.dropna()
# Loại bỏ trùng lặp
df = df.drop_duplicates(subset=['url'])
# Chuẩn hóa (về chữ thường)
df['url'] = df['url'].str.lower()

# 3. TRÍCH XUẤT ĐẶC TRƯNG (Feature Extraction)
def extract_features(url):
    features = {}
    # Đặc trưng 1: Độ dài URL
    features['url_length'] = len(url)
    # Đặc trưng 2: Số dấu chấm
    features['count_dots'] = url.count('.')
    # Đặc trưng 3: Kiểm tra xem có dùng IP không
    features['is_ip'] = 1 if re.match(r'.*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}.*', url) else 0
    # Đặc trưng 4: Số dấu gạch ngang (phishing hay dùng cái này)
    features['count_hyphens'] = url.count('-')
    # Đặc trưng 5: Có từ khóa nhạy cảm không
    sensitive_words = ['login', 'verify', 'secure', 'bank', 'update']
    features['has_sensitive_word'] = 1 if any(word in url for word in sensitive_words) else 0
    
    return pd.Series(features)

# Áp dụng trích xuất vào DataFrame
feature_df = df['url'].apply(extract_features)
final_df = pd.concat([df, feature_df], axis=1)

print("\n--- Dữ liệu sau khi Làm sạch & Trích xuất ---")
print(final_df)

# 4. CHUẨN BỊ DỮ LIỆU TRAIN/TEST
X = final_df.drop(['url', 'label'], axis=1) # Các đặc trưng (số)
y = final_df['label']                       # Nhãn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nKích thước tập Train: {X_train.shape}")
print(f"Kích thước tập Test: {X_test.shape}")

# Lưu kết quả ra file CSV để dùng cho tuần sau (Training)
final_df.to_csv('cleaned_dataset.csv', index=False)
print("\nĐã lưu dữ liệu sạch vào file: cleaned_dataset.csv")
