# Стадия для сборки фронтенда
FROM node:18 as frontend-builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY . .
RUN npm run build

# Стадия для бэкенда
FROM python:3.9-slim as backend
WORKDIR /app

# Установка зависимостей бэкенда
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем код бэкенда
COPY app.py .

# Копируем собранный фронтенд в папку, доступную для Flask
COPY --from=frontend-builder /app/dist /app/static

# Указываем Flask, где искать статические файлы
ENV FLASK_STATIC_FOLDER=/app/static

# Команда для запуска бэкенда
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
