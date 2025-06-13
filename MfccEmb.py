import os
import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import umap
import torch
from speechbrain.inference import EncoderClassifier

# -----------------------------
# 1. Загрузка данных
# -----------------------------
data_dir = "16000_pcm_speeches"

speakers = [s for s in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, s))
            and s not in ["background_noise", "other", "_background_noise_", ".DS_Store"]
            and len([f for f in os.listdir(os.path.join(data_dir, s)) if f.endswith(".wav")]) >= 2]
print("🎤 Найдено спикеров:", len(speakers))

train_file_paths, train_labels = [], []
test_file_paths, test_labels = [], []

for speaker in speakers:
    folder = os.path.join(data_dir, speaker)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    full_paths = [os.path.join(folder, f) for f in files]
    labels = [speaker] * len(full_paths)
    if len(full_paths) >= 2:
        train, test, y_train_split, y_test_split = train_test_split(
            full_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        train_file_paths.extend(train)
        train_labels.extend(y_train_split)
        test_file_paths.extend(test)
        test_labels.extend(y_test_split)

# Проверка количества файлов
for speaker in speakers:
    folder = os.path.join(data_dir, speaker)
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    print(f"Спикер {speaker}: {len(files)} файлов")

# проверка шумов
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

print("Уникальные классы в y_train:", np.unique(train_labels))
print("Уникальные классы в y_test:", np.unique(test_labels))

# -----------------------------
# 2. Извлечение speaker embeddings
# -----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🛠 Используем устройство: {device}")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)


def extract_speaker_embedding(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
        embedding = classifier.encode_batch(signal)
        return embedding.squeeze().cpu().detach().numpy()
    except Exception as e:
        print(f"⚠️ Ошибка при обработке {file_path}: {e}")
        return None

def extract_embeddings_batch(paths, labels):
    embeddings, valid_labels = [], []
    for path, label in zip(paths, labels):
        emb = extract_speaker_embedding(path)
        if emb is not None:
            embeddings.append(emb)
            valid_labels.append(label)
    return np.array(embeddings), np.array(valid_labels)

print("🔄 Извлечение эмбеддингов для тренировочной выборки...")
X_train_raw, y_train = extract_embeddings_batch(train_file_paths, train_labels)
print("🔄 Извлечение эмбеддингов для тестовой выборки...")
X_test_raw, y_test = extract_embeddings_batch(test_file_paths, test_labels)

print(f"Обработано тренировочных файлов: {len(X_train_raw)} из {len(train_file_paths)}")
print(f"Обработано тестовых файлов: {len(X_test_raw)} из {len(test_file_paths)}")

# Фильтрация тренировочного набора
valid_classes = np.unique(y_test)
mask = np.isin(y_train, valid_classes)
X_train_raw = X_train_raw[mask]
y_train = y_train[mask]
print("Уникальные классы в y_train после фильтрации:", np.unique(y_train))


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

print(f"Размер X_train: {X_train.shape}")
print(f"Размер X_test: {X_test.shape}")

# -----------------------------
# 5. kNN с использованием sklearn
# -----------------------------
fixed_k = 26
knn = KNeighborsClassifier(n_neighbors=fixed_k, metric="cosine")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Уникальные предсказанные классы:", np.unique(y_pred))

# -----------------------------
# 6. Отчеты
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Точность модели на тестовой выборке: {accuracy:.4f}")
print("\n🔍 Классификационный отчет:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 7. Визуализация: t-SNE / UMAP для 2D и 3D
# -----------------------------
print("\n📊 Начинаем визуализацию: t-SNE и UMAP...")

# Объединяем train + test для визуализации
X_combined = np.vstack([X_train_norm, X_test_norm])
y_combined = np.hstack([y_train, y_test])

# t-SNE 2D
print("🧮 Применяю t-SNE 2D...")
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne_2d = tsne_2d.fit_transform(X_combined)

# t-SNE 3D
print("🧮 Применяю t-SNE 3D...")
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
X_tsne_3d = tsne_3d.fit_transform(X_combined)

# UMAP 2D
print("🧮 Применяю UMAP 2D...")
umap_2d = umap.UMAP(n_components=2, random_state=42)
X_umap_2d = umap_2d.fit_transform(X_combined)

# UMAP 3D
print("🧮 Применяю UMAP 3D...")
umap_3d = umap.UMAP(n_components=3, random_state=42)
X_umap_3d = umap_3d.fit_transform(X_combined)

# Добавляем метки в DataFrame
df = pd.DataFrame({
    'Label': y_combined,
    'tSNE1': X_tsne_2d[:, 0],
    'tSNE2': X_tsne_2d[:, 1],
    'tSNE3': X_tsne_3d[:, 2],
    'UMAP1': X_umap_2d[:, 0],
    'UMAP2': X_umap_2d[:, 1],
    'UMAP3': X_umap_3d[:, 2]
})

# --- 2D t-SNE график ---
plt.figure(figsize=(10, 8))
sns.scatterplot(x='tSNE1', y='tSNE2', hue='Label', data=df, palette='tab20', legend='full', s=60)
plt.title('t-SNE 2D: Распределение классов по эмбеддингам')
plt.xlabel('tSNE1')
plt.ylabel('tSNE2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig('tsne_2d_speaker_embeddings.png')
plt.close()

# --- 2D UMAP график ---
plt.figure(figsize=(10, 8))
sns.scatterplot(x='UMAP1', y='UMAP2', hue='Label', data=df, palette='tab20', legend='full', s=60)
plt.title('UMAP 2D: Распределение классов по эмбеддингам')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.savefig('umap_2d_speaker_embeddings.png')
plt.close()

# --- 3D t-SNE график с Plotly ---
fig = px.scatter_3d(df, x='tSNE1', y='tSNE2', z='tSNE3',
                    color='Label', title='t-SNE 3D: Распределение классов',
                    hover_data=['Label'])
fig.update_traces(marker=dict(size=4))
fig.write_html('tsne_3d_speaker_embeddings.html')

# --- 3D UMAP график с Plotly ---
fig = px.scatter_3d(df, x='UMAP1', y='UMAP2', z='UMAP3',
                    color='Label', title='UMAP 3D: Распределение классов',
                    hover_data=['Label'])
fig.update_traces(marker=dict(size=4))
fig.write_html('umap_3d_speaker_embeddings.html')

print("✅ Графики сохранены:")
print(" - tsne_2d_speaker_embeddings.png")
print(" - umap_2d_speaker_embeddings.png")
print(" - tsne_3d_speaker_embeddings.html")
print(" - umap_3d_speaker_embeddings.html")

# -----------------------------
# 8. Матрица ошибок
# -----------------------------
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title(f"Матрица ошибок (k = {fixed_k})")
plt.xlabel("Предсказано")
plt.ylabel("Истинно")
plt.tight_layout()
plt.savefig("confusion_matrix_speaker_embeddings.png")
print("📊 Матрица ошибок сохранена как confusion_matrix_speaker_embeddings.png")
plt.close()

# -----------------------------
# 9. Тестирование на своём файле
# -----------------------------
def test_custom_file(file_path):
    print(f"\n🔎 Тестирование на файле: {file_path}")
    try:
        embedding = extract_speaker_embedding(file_path)
        if embedding is not None:
            embedding = normalize(embedding, X_train_mean, X_train_std)
            embedding = pca.transform([embedding])
            prediction = knn.predict(embedding)
            print(f"📢 Предсказанный класс: {prediction[0]}")
        else:
            print("❌ Не удалось извлечь эмбеддинг")
    except Exception as e:
        print(f"❌ Ошибка при обработке файла: {e}")

test_custom_file("/Users/tagirahmetsin/Downloads/g2.wav")

# -----------------------------
# 10. Очистка временных файлов
# -----------------------------
for path in temp_noise_paths:
    try:
        os.remove(path)
    except:
        pass
