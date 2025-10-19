import cv2
import numpy as np
import os

input_folder = "cam"
output_folder = "urun_kisimlari_no_bg"
os.makedirs(output_folder, exist_ok=True)

# Beyaz renk aralığı (genelde açık ve düşük saturasyon)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{filename} okunamadı!")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Beyaz arka plan maskesi
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Beyaz alanları maskeden çıkar (arka planı siyah yap)
        mask_inv = cv2.bitwise_not(mask_white)

        # Görüntünün sadece ürün kısmını al
        result = cv2.bitwise_and(img, img, mask=mask_inv)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, result)
        print(f"{filename} işlendi, arka plan temizlendi.")
