<template>
  <div class="dictionary-manager">
    <div class="top-left">
      <router-link to="/">
        <img src="/left-arrow.svg" alt="Back to Home" class="icon-button" />
      </router-link>
    </div>
    <h1>Управление словарём</h1>
    <!-- Форма для добавления слова -->
    <input type="text" v-model="word" placeholder="Введите слово" />
    <div class="add-button-container">
      <button @click="addWord" :disabled="!word.trim()">
        Добавить слово
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
        } else {
          const error = await response.json();
          console.error("Ошибка при добавлении слова:", error);
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
  width: 80%;
}

/* Контейнер для кнопки добавления – центрирование */
.add-button-container {
  margin: 20px auto;
  display: flex;
  justify-content: center;
}

.add-button-container button {
  padding: 10px 20px;
  background-color: #42b983; /* Зелёный фон */
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Состояние, когда кнопка отключена */
.add-button-container button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  color: #666;
}
</style>