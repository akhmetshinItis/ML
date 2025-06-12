import os
import librosa
import numpy as np
from pydub import AudioSegment
from collections import Counter
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

# -----------------------------
# 1. Загрузка данных
# -----------------------------
data_dir = "/Users/tagirahmetsin/Downloads/16000_pcm_speeches"
speakers = [s for s in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, s)) and s not in ["background_noise", "other", "_background_noise_", ".DS_Store"]]
print("Список спикеров:", speakers)

train_file_paths, train_labels = [], []
test_file_paths, test_labels = [], []

for speaker in speakers:
    folder = os.path.join(data_dir, speaker)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    full_paths = [os.path.join(folder, f) for f in files]
    labels = [speaker] * len(full_paths)
    train, test, y_train_split, y_test_split = train_test_split(
        full_paths, labels, test_size=0.4, random_state=42)
    train_file_paths.extend(train)
    train_labels.extend(y_train_split)
    test_file_paths.extend(test)
    test_labels.extend(y_test_split)

# -----------------------------
# 1.2 Добавление шумов
# -----------------------------
background_noise_dir = os.path.join(data_dir, "_background_noise_")
temp_noise_paths = []
if os.path.exists(background_noise_dir):
    for file in os.listdir(background_noise_dir):
        if file.endswith(".wav"):
            audio = AudioSegment.from_wav(os.path.join(background_noise_dir, file))
            for i, start in enumerate(range(0, len(audio), 1000)):
                end = min(start + 1000, len(audio))
                chunk = audio[start:end]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    chunk.export(tmp.name, format="wav")
                    train_file_paths.append(tmp.name)
                    train_labels.append("noise")
                    temp_noise_paths.append(tmp.name)

# -----------------------------
# 2. Извлечение признаков
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    combined = np.vstack([mfcc, delta, delta2, chroma, contrast, zcr, rms])
    return np.hstack([np.mean(combined, axis=1),
                      np.std(combined, axis=1),
                      np.median(combined, axis=1)])

def extract_features_batch(paths, labels):
    feats, valid_labels = [], []
    for path, label in zip(paths, labels):
        try:
            feats.append(extract_features(path))
            valid_labels.append(label)
        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")
    return np.array(feats), np.array(valid_labels)

X_train_raw, y_train = extract_features_batch(train_file_paths, train_labels)
X_test_raw, y_test = extract_features_batch(test_file_paths, test_labels)

# -----------------------------
# 3. Нормализация
# -----------------------------
def normalize(X, mean, std):
    return (X - mean) / (std + 1e-8)

X_train_mean = X_train_raw.mean(axis=0)
X_train_std = X_train_raw.std(axis=0)
X_train_norm = normalize(X_train_raw, X_train_mean, X_train_std)
X_test_norm = normalize(X_test_raw, X_train_mean, X_train_std)

# -----------------------------
# 4. PCA
# -----------------------------
pca = PCA(n_components=30)
X_train = pca.fit_transform(X_train_norm)
X_test = pca.transform(X_test_norm)

# -----------------------------
# 5. kNN
# -----------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k=5):
    distances = [(euclidean_distance(x_test, x), y) for x, y in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in k_neighbors]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0]

fixed_k = 26
print(f"\nИспользуется фиксированное значение k = {fixed_k}")
y_pred = [knn_predict(X_train, y_train, x, k=fixed_k) for x in X_test]

# -----------------------------
# 6. Отчёт и визуализация
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title(f"Матрица ошибок (k = {fixed_k})")
plt.xlabel("Предсказано")
plt.ylabel("Истинно")
plt.tight_layout()
plt.savefig("confusion_matrix_kNN.png")
print("Матрица ошибок сохранена как confusion_matrix_kNN.png")
plt.close()

# -----------------------------
# 7. Тестирование на своём файле
# -----------------------------
def test_custom_file(file_path):
    print(f"\nТестирование на файле: {file_path}")
    try:
        features = extract_features(file_path)
        features = normalize(features, X_train_mean, X_train_std)
        features = pca.transform([features])
        prediction = knn_predict(X_train, y_train, features[0], k=fixed_k)
        print(f"Предсказанный класс: {prediction}")
    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")

test_custom_file("/Users/tagirahmetsin/Downloads/g2.wav")

# -----------------------------
# 8. Очистка временных файлов
# -----------------------------
for path in temp_noise_paths:
    try:
        os.remove(path)
    except Exception as e:
        print(f"Ошибка при удалении временного файла: {e}")