import os
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l2
    import matplotlib.pyplot as plt
    from tensorflow.keras.datasets import imdb
except ModuleNotFoundError as e:
    print(
        "Ошибка: Необходимые модули (Keras или TensorFlow) не установлены. Установите их с помощью 'pip install tensorflow'.")
    raise e

# Установка конфигурации TensorFlow для оптимизации потоков
tf.config.threading.set_intra_op_parallelism_threads(8)  # Количество потоков для операций
tf.config.threading.set_inter_op_parallelism_threads(4)  # Количество потоков для взаимодействия

# Загрузка стоп-слов
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Предобработка текста
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Добавление дополнительных отрицательных примеров
def augment_negative_data(reviews, labels):
    augmented_reviews = reviews.copy()
    augmented_labels = labels.copy()
    negative_phrases = [
        "never again", "absolute disaster", "horrific mistake", "worst ever", "not worth it",
        "utterly disappointing", "failed miserably", "do not recommend", "nothing redeeming about it"
    ]
    for phrase in negative_phrases:
        augmented_reviews.append(phrase)
        augmented_labels.append(0)  # 0 для отрицательной тональности
    return augmented_reviews, augmented_labels

# Визуализация графиков истории обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # Сохранение графика в файл
    plt.savefig('training_history.png')
    print("Графики сохранены в 'training_history.png'")

### Обучение и сохранение модели
def train_and_save_model():
    if os.path.exists('sentiment_model_demo.h5') and os.path.exists('tokenizer_demo.pkl'):
        print("Модель и токенизатор уже существуют. Пропускаем обучение.")
        return

    # Загрузка датасета IMDb
    max_features = 20000  # Максимальное количество уникальных слов
    maxlen = 200  # Максимальная длина текстов

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

    # Преобразование последовательностей в текст для предобработки
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    def decode_review(sequence):
        return ' '.join([reverse_word_index.get(i - 3, "?") for i in sequence])

    train_texts = [decode_review(sequence) for sequence in train_data]
    test_texts = [decode_review(sequence) for sequence in test_data]

    # Очистка данных
    train_texts_cleaned = [clean_text(text) for text in train_texts]
    test_texts_cleaned = [clean_text(text) for text in test_texts]

    # Расширение данных с отрицательной полярностью
    train_texts_cleaned, train_labels = augment_negative_data(train_texts_cleaned, list(train_labels))

    # Создание токенизатора
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_texts_cleaned)

    # Преобразование текстов в последовательности
    train_sequences = tokenizer.texts_to_sequences(train_texts_cleaned)
    test_sequences = tokenizer.texts_to_sequences(test_texts_cleaned)
    train_sequences_padded = pad_sequences(train_sequences, maxlen=maxlen)
    test_sequences_padded = pad_sequences(test_sequences, maxlen=maxlen)

    # Сохранение токенизатора
    with open('tokenizer_demo.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # Использование предобученных эмбеддингов GloVe
    embeddings_index = {}
    try:
        with open('glove.6B.300d.txt', encoding='utf-8') as glove_file:
            for line in glove_file:
                values = line.split()
                word = values[0]
                coefficients = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefficients
    except FileNotFoundError:
        print("Ошибка: Файл 'glove.6B.300d.txt' не найден. Проверьте наличие файла в каталоге.")
        return

    embedding_dim = 300
    embedding_matrix = np.zeros((max_features, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_features:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    # Построение модели
    model = models.Sequential([
        layers.Embedding(max_features, embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=True),
        layers.Bidirectional(layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Автоматическое определение весов классов
    class_weights = {0: len(train_labels) / (2 * np.bincount(train_labels)[0]),
                     1: len(train_labels) / (2 * np.bincount(train_labels)[1])}

    # Добавление EarlyStopping для предотвращения переобучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Обучение модели
    history = model.fit(
        train_sequences_padded,
        np.array(train_labels),
        epochs=50,  # Уменьшенное количество эпох для снижения нагрузки
        batch_size=64,  # Умеренный размер батча для процессоров
        class_weight=class_weights,
        validation_data=(test_sequences_padded, np.array(test_labels)),
        callbacks=[early_stopping]
    )

    # Отображение графиков обучения
    plot_training_history(history)

    # Сохранение модели
    model.save('sentiment_model_demo.h5')
    print("Модель сохранена как 'sentiment_model_demo.h5'")

### Использование модели для предсказания
def predict_sentiment_demo(reviews):
    # Проверка наличия файлов
    if not os.path.exists('sentiment_model_demo.h5') or not os.path.exists('tokenizer_demo.pkl'):
        print("Модель или токенизатор отсутствуют. Запуск обучения...")
        train_and_save_model()

    # Загрузка модели и токенизатора
    try:
        model = load_model('sentiment_model_demo.h5')
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    try:
        with open('tokenizer_demo.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("Токенизатор успешно загружен.")
    except Exception as e:
        print(f"Ошибка загрузки токенизатора: {e}")
        return

    # Очистка и токенизация текстов
    reviews_cleaned = [clean_text(review) for review in reviews]
    sequences = tokenizer.texts_to_sequences(reviews_cleaned)
    padded_sequences = pad_sequences(sequences, maxlen=200)

    # Прогнозирование
    try:
        predictions = model.predict(padded_sequences)
        for review, prediction in zip(reviews, predictions):
            sentiment = "Положительная" if prediction > 0.5 else "Отрицательная"
            probability = prediction[0] if sentiment == "Положительная" else 1 - prediction[0]
            print(f"Рецензия: '{review}'\nТональность: {sentiment} (вероятность: {probability:.2f})\n")
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")

### Основной блок
if __name__ == "__main__":
    # Пример использования модели
    if not os.path.exists('sentiment_model_demo.h5'):
        train_and_save_model()

    example_reviews = [
        "I absolutely loved this movie!",
        "This was the worst film I have ever seen.",
        "The storyline was captivating and thrilling.",
        "I would never recommend this to anyone.",
        "A breathtaking and emotional journey.",
        "Completely overrated and disappointing.",
        "One of the best films in recent years.",
        "Terrible plot and poorly written characters.",
        "An absolute disaster of a movie.",
        "Horrific pacing and a boring story.",
        "Fantastic visuals but no substance.",
        "A masterpiece that left me speechless.",
        "Worst acting I've seen in years.",
        "Brilliant soundtrack but nothing else stood out.",
        "A waste of time and money.",
        "Truly inspiring and heartwarming experience.",
        "The direction was confusing and messy.",
        "An epic story told beautifully.",
        "Nothing worked in this film, complete flop.",
        "Absolutely stunning visuals and plot."
    ]
    predict_sentiment_demo(example_reviews)
