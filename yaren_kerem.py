from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Görüntü dosyalarını yükleme
data_dir = '/content/drive/MyDrive/DeusAIImageDetection/ChessPieces'

# Görüntü boyutu ayarları
image_size = (128, 128)

# Örnek görüntü yükleme fonksiyonu
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)#bu adım bir normalizasyon adımıdır
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Alt klasörlerin bulunması için
subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()] #os.scandir(data_dir) alt klasörlerin bulunmasını sağlar.
# Alt klasörlerin isimlerini almak için
subfolder_names = [os.path.basename(f.path) for f in os.scandir(data_dir) if f.is_dir()]

print(f"Alt klasörler: {subfolder_names}")

# Her bir alt klasörden örnek görüntüler yükleyip gösterme
for subfolder in subfolders:
    sample_images = os.listdir(subfolder)[:1]  # İlk görüntüyü al
    for img_name in sample_images:
        img_path = os.path.join(subfolder, img_name)
        img = load_image(img_path)
        plt.imshow(img)
        plt.title(f"Class: {os.path.basename(subfolder)}")
        plt.axis('off') #eksen numaraları görünmemesi için
        plt.show()

"""!!!!!Model performansına göre hangilerinin kullanılacağı
değişiklik gösterir.!!!!!"""
#DATA CLEANING
def clean_image(image):
    # Gürültü azaltma
    denoised_img = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Keskinleştirme
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    sharpened_img = cv2.filter2D(denoised_img, -1, kernel)

    # HSV renk uzayına dönüştürme
    hsv_img = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2HSV)

    # Mavi renk filtresi
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    blue_highlighted_img = cv2.bitwise_and(sharpened_img, sharpened_img, mask=blue_mask)

    # Kontrast ve parlaklık ayarları
    adjusted_img = cv2.convertScaleAbs(sharpened_img, alpha=1.3, beta=20)

    # Histogram eşitleme
    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    hist_eq_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)

    # Kenar belirginleştirme
    edges = cv2.Canny(equalized_img, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_highlighted_img = cv2.addWeighted(sharpened_img, 0.7, edges_colored, 0.3, 0)

    # Renk dengesini ayarlama
    b, g, r = cv2.split(edge_highlighted_img)
    r = cv2.add(r, 10)
    b = cv2.subtract(b, 10)
    balanced_img = cv2.merge([b, g, r])

    # Sonuçları birleştirme
    final_img = cv2.addWeighted(balanced_img, 0.8, blue_highlighted_img, 0.2, 0)

    return final_img

# Temizlenmiş görüntüleri gösterme
for subfolder in subfolders:
    sample_images = os.listdir(subfolder)[:1]  # İlk görüntüyü al
    for img_name in sample_images:
        img_path = os.path.join(subfolder, img_name)
        img = load_image(img_path)
        cleaned_img = clean_image(img)  # Görüntüyü temizle

        # Orijinal ve temizlenmiş görüntüyü yan yana göster
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Orijinal Görüntü')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cleaned_img)
        plt.title('Temizlenmiş Görüntü')
        plt.axis('off')
        plt.show()

resim_yolu= "/content/drive/MyDrive/DeusAIImageDetection/ChessPieces/train"

image_files = [f for f in os.listdir(resim_yolu) if os.path.isfile(os.path.join(resim_yolu, f))]

print(f"Resim isimleri: {image_files}")

resimler = []
for image_name in image_files:
    image_path = os.path.join(resim_yolu, image_name)
    img = load_image(image_path) #resmi yükledim
    resimler.append(img)


images_array = np.array(resimler)

print(images_array.shape) #yükledigim resimlerin boyutları

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,                 #VERİ ARTTIRMA FONKSİYONU BU KOD SAYESİNDE OLUYOR
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def gorsellestirme(data_generator, images, num_augmented_images=5):
    for idx, image in enumerate(images):

        plt.figure(figsize=(15, 5))

        sample_image = image.reshape((1, image_size[0], image_size[1], 3))
        image_iterator = data_generator.flow(sample_image, batch_size=1)

        for j in range(num_augmented_images):
            augmented_image = image_iterator.__next__()[0].astype('uint8')

            plt.subplot(1, num_augmented_images, j + 1)
            plt.imshow(augmented_image)
            plt.axis('off')
            if j == 0:
                plt.title(f"Image {idx + 1}")

        plt.show()

gorsellestirme(datagen,resimler, num_augmented_images=5)