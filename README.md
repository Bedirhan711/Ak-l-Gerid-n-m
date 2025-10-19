# ♻️ Derin Öğrenme Tabanlı Geri Dönüşüm Sistemi

## 📘 Proje Tanımı
Bu proje, geri dönüşüm sürecinde sıkça karşılaşılan hatalı atık ayrıştırma problemini çözmek amacıyla geliştirilmiş **derin öğrenme tabanlı otomatik geri dönüşüm sistemi**dir. İnsan hatalarına açık olan manuel ayrıştırma işlemleri yerine, **yapay zekâ** ve **görüntü işleme** teknikleri kullanılarak atıklar **plastik**, **metal** ve **cam** olarak otomatik biçimde sınıflandırılır. Böylece geri dönüşüm süreçleri daha **doğru**, **hızlı** ve **verimli** hale getirilir.

---

## 🧠 Kullanılan Yöntemler ve Araçlar

- **Derin Öğrenme Mimarileri:**  
  - ResNet34  
  - DenseNet121  
  - EfficientNet-B1  

- **Kütüphaneler ve Ortamlar:**  
  - PyTorch  
  - Visual Studio Code  

- **Model Eğitimi Parametreleri:**  
  - Epoch Sayısı: 15  
  - Batch Size: 8  
  - Öğrenme Oranı (Learning Rate): 0.001  
  - Optimizasyon Algoritması: Adam  
  - Early Stopping yöntemiyle overfitting engellenmiştir.  

---

## ⚙️ Donanım ve Sistem Yapısı

- **İşletim Sistemi:** Windows 11  
- **Mikrodenetleyici:** Arduino Uno  
- **Motor Kontrol Kartı:** PCA9685 (I2C haberleşme)  
- **Servo Motor:** 4 adet  
- **Görüntü Alma Birimi:** Webcam  

Bu donanım bileşenleri sayesinde modelden alınan sınıflandırma sonuçlarına göre servo motorlar hareket eder ve atıklar ilgili bölmelere yönlendirilir.

---

## 📊 Bulgular ve Sonuçlar

Model eğitimi sonucunda elde edilen performans, üç temel metrik ile analiz edilmiştir:

- **Kayıp Grafiği (Loss Curve):** Modelin hata oranının eğitim boyunca azaldığı gözlemlenmiştir.  
- **Karmaşıklık Matrisi (Confusion Matrix):** Sınıflar bazında doğru ve yanlış tahmin sayıları incelenmiştir.  
- **F1-Skoru:** Modelin **precision** ve **recall** dengesi değerlendirilmiştir.  

Gerçek dünya koşullarında, özellikle ışık farkları ve benzer görünümlü atıkların (örneğin saydam plastik ve cam şişeler) model performansını etkileyebileceği belirlenmiştir.

---

## 🧩 Sonuç

Bu çalışma, **geri dönüşüm kutularında atık ayrıştırma işlemini akıllı hale getiren**, çevre dostu ve yenilikçi bir sistem ortaya koymaktadır. Derin öğrenme mimarilerinin kullanımıyla yüksek doğruluk oranına ulaşılmış; donanım entegrasyonu sayesinde teorik model pratik bir çözüme dönüştürülmüştür.  

---

## 👥 Proje Ekibi
- **Mehmet Salih Bedirhanoğlu**  
- **Mahmut Eren Zerdeci**

---

## 📚 Kaynakça
- Chandini (2020). *ResNet (34, 50, 101)… what actually it is?* Medium.  
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR. [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)  
- LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11): 2278–2324.  

---

📌 *Bu proje, derin öğrenme yöntemleriyle sürdürülebilir çevre teknolojileri alanına katkı sağlamayı amaçlamaktadır.*
