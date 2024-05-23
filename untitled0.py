import shutil
from pathlib import Path

# Kaynak ve hedef klasörlerin tanımlanması
data_path= Path('input')
input_path= Path('Data')

# Sınıfların listesi
folders = ['glioma_tumor', 'meningioma_tumor', 'normal', 'pituitary_tumor']



# Etiket dosyasının kopyalanması
shutil.copy(input_path / 'trainLabels.csv', data_path/ 'trainLabels.csv')

# Eğitim ve test klasörlerinin adlarını belirleme
train_folder = data_path / 'train'
test_folder = data_path / 'test'

# Eğitim ve test klasörlerini yeniden adlandırma
train_folder.rename(data_path / 'original_train')
test_folder.rename(data_path / 'original_test')

# Klasörlerin başarıyla kopyalandığı ve yeniden adlandırıldığına dair mesaj
print("Klasörler başarıyla kopyalandı ve yeniden adlandırıldı.")
