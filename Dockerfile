# Стадия для сборки фронтенда
FROM node:18 as frontend-builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

# Стадия для бэкенда
FROM python:3.9 as backend
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Установка зависимостей бэкенда
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем файл модели
COPY letter_recognition_model.h5 .

# Копируем файл словаря
COPY cleaned_filtered_russian_words.json .

# Копируем код бэкенда
COPY app.py .

# Копируем собранный фронтенд в папку static
COPY --from=frontend-builder /app/dist /app/static

# Указываем Flask, где искать статические файлы
ENV FLASK_STATIC_FOLDER=/app/static

# Команда для запуска бэкенда
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]