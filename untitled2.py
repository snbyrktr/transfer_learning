import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Klasörlerin adları
folders = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']

# Veri ve etiket listeleri
data = []
labels = []

# Görselleri gösterme boyutu
img_size = (128, 128)

# Her klasörde dolaşma ve veri toplama
for folder_ in folders:
    fold_path = os.path.join('Data', folder_)  # 'Data' klasöründe olduğunu varsayalım
    lab_val = folders.index(folder_)
    for i, dosya in enumerate(os.listdir(fold_path)):
        if i >= 100:  # Her klasörden maksimum 100 görsel alınacak
            break
        dosya_yolu = os.path.join(fold_path, dosya)
        img = cv2.imread(dosya_yolu)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)
        labels.append(lab_val)

# Veri ve etiket listelerini numpy dizilerine dönüştürme
data = np.array(data)
labels = np.array(labels)

# Verileri eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# VGG16 modelini yükleme
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Evrişim katmanlarını dondurma
for layer in vgg.layers:
    layer.trainable = False

# Yeni bir model oluşturma
model = Sequential([
    vgg,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Sınıf sayısı: 4
])

# Modelin katmanlarını gösterme
print("Model Katmanları:")
for i, layer in enumerate(model.layers):
    print(f"{i+1}. Katman: {layer.name} - Trainable: {layer.trainable}")

# Modeli derleme
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Modelin değerlendirilmesi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")









#################################
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Veriyi normalize etme
data = data / 255.0

# Etiketleri one-hot encode etme
labels = to_categorical(labels, num_classes=4)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# VGG16 modelini yükleme ve son katmanlarını değiştirme
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = Flatten()(vgg.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=output)

# Son katmanların eğitilebilir olduğunu ayarlama
for layer in vgg.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Modeli değerlendirme
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Eğitim ve doğrulama kaybını ve doğruluğunu görselleştirme
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()