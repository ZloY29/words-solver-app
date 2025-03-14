<template>
    <div class="dictionary-manager">
      <div class="top-left">
        <router-link to="/">
          <img src="/left-arrow.svg" alt="Back to Home" class="icon-button" />
        </router-link>
      </div>
      <h1>Управление словарём</h1>
      <!-- Форма для добавления/удаления слова -->
      <input type="text" v-model="word" placeholder="Введите слово" />
      <div class="btn-group">
        <!-- Кнопка добавления -->
        <button @click="addWord" :disabled="!word.trim()">
          Добавить слово
        </button>
        <!-- Кнопка удаления -->
        <button @click="deleteWord" :disabled="!word.trim()">
          Удалить слово
        </button>
      </div>
      <br>
    </div>
  </template>
  
  <script>
  export default {
    name: 'DictionaryManager',
    data() {
      return {
        word: '',
      };
    },
    methods: {
      async addWord() {
        if (!this.word.trim()) return;
        try {
          const response = await fetch('/add_word', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ word: this.word.trim() }),
          });
          if (response.ok) {
            const data = await response.json();
            console.log("Слово добавлено:", data);
            this.word = ''; // очищаем поле ввода
            // Можно также обновить локальный список слов, если он есть
          } else {
            const error = await response.json();
            console.error("Ошибка при добавлении слова:", error);
          }
        } catch (error) {
          console.error("Ошибка сети:", error);
        }
      },
      async deleteWord() {
        if (!this.word.trim()) return;
        try {
          const response = await fetch('/remove_word', {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ word: this.word.trim() }),
          });
          if (response.ok) {
            const data = await response.json();
            console.log("Слово удалено:", data);
            this.word = ''; // очищаем поле ввода
            // Можно также обновить локальный список слов, если он есть
          } else {
            const error = await response.json();
            console.error("Ошибка при удалении слова:", error);
          }
        } catch (error) {
          console.error("Ошибка сети:", error);
        }
      },
    },
  };
  </script>
  
  <style scoped>
  .top-left {
    position: absolute;
    top: 20px;
    left: 20px;
  }
  .icon-button {
    width: 32px;
    height: 32px;
    cursor: pointer;
  }
  .dictionary-manager {
    text-align: center;
    margin-top: 50px;
  }
  input[type="text"] {
    padding: 10px;
    margin: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
  }
  .btn-group button {
    padding: 10px 20px;
    background-color: #42b983; /* Зелёный фон */
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    margin: 0 10px;
  }

  /* Состояние, когда кнопка отключена */
  .btn-group button:disabled {
    background-color: #ccc;     /* Серый фон */
    cursor: not-allowed;        /* Запрещающий курсор */
    color: #666;                /* Можно затемнить текст */
  }
  .btn-group {
    display: flex;
    flex-direction: row; /* по умолчанию в ряд */
    justify-content: center;
    gap: 10px;
  }
  /* Для экранов уже 600px переключаемся в столбец */
  @media (max-width: 600px) {
    .btn-group {
      flex-direction: column;
      align-items: center;
    }
  }
  </style>