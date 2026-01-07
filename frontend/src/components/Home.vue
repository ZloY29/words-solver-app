<template>
  <div class="home">
    <!-- Левая верхняя кнопка: либо «мусорка» (войти в режим), либо «Отмена» (выйти) -->
    <div class="top-left">
      <template v-if="!massDeleteMode">
        <!-- Кнопка без заливки -->
        <button class="icon-button-plain" @click="enterMassDeleteMode">
          <img src="/trash-icon.svg" alt="Mass Delete" class="icon-button" />
        </button>
      </template>
      <template v-else>
        <button class="cancel-button" @click="exitMassDeleteMode">
          Отмена
        </button>
      </template>
    </div>

    <!-- Кнопка перехода к управлению словарём -->
    <div class="top-right">
      <router-link to="/manage">
        <img src="/book.svg" alt="Manage Dictionary" class="icon-button" />
      </router-link>
    </div>

    <h1>Photo Word Finder</h1>

    <input type="file" accept="image/*" @change="handleFileChange" />
    <button @click="uploadImage" :disabled="!selectedImage">Upload Image</button>

    <div v-if="loading">Loading...</div>

    <div v-if="results.length > 0">
      <h2>Found Words:</h2>
      <ul>
        <li v-for="(word, index) in results" :key="index">
          <!-- Чекбокс показываем только если включен massDeleteMode -->
          <template v-if="massDeleteMode">
            <input
              type="checkbox"
              v-model="selectedWords"
              :value="word.name"
            />
          </template>
          {{ word.name }} - {{ word.score }} points
        </li>
      </ul>

      <!-- Кнопка «Удалить выбранные» теперь внизу под списком -->
      <div v-if="massDeleteMode" class="mass-delete-bottom">
        <button @click="deleteSelectedWords" class="delete-button">
          Удалить выбранные
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Home',
  data() {
    return {
      selectedImage: null,
      results: [],
      loading: false,
      massDeleteMode: false,
      selectedWords: [],
    };
  },
  methods: {
    handleFileChange(event) {
      this.selectedImage = event.target.files[0];
    },
    async uploadImage() {
      if (!this.selectedImage) return;
      this.loading = true;
      const formData = new FormData();
      formData.append("image", this.selectedImage);
      try {
        const response = await fetch("/upload", {
          method: "POST",
          body: formData,
        });
        if (response.ok) {
          const data = await response.json();
          this.results = data.words;
        } else {
          console.error("Error uploading image");
        }
      } catch (error) {
        console.error("Network error:", error);
      } finally {
        this.loading = false;
      }
    },
    enterMassDeleteMode() {
      this.massDeleteMode = true;
    },
    exitMassDeleteMode() {
      this.massDeleteMode = false;
      this.selectedWords = [];
    },
    async deleteSelectedWords() {
      if (!this.selectedWords.length) return;
      try {
        const response = await fetch('/remove_words', {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ words: this.selectedWords })
        });
        if (response.ok) {
          const data = await response.json();
          console.log(data.message);
          // Обновляем список слов, исключая удалённые
          this.results = this.results.filter(
            wordObj => !this.selectedWords.includes(wordObj.name)
          );
          this.exitMassDeleteMode();
        } else {
          console.error("Ошибка при удалении слов:", response.statusText);
        }
      } catch (error) {
        console.error("Ошибка сети:", error);
      }
    },
  },
};
</script>

<style scoped>
.home {
  text-align: center; /* Центрируем содержимое */
  margin-top: 50px;
}

/* Убираем лишнюю заливку у кнопки с иконкой */
.icon-button-plain {
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}

/* Иконки по-прежнему можно масштабировать через класс */
.icon-button {
  width: 32px;
  height: 32px;
  cursor: pointer;
}

/* Кнопки массовых действий */
.mass-actions button,
.mass-delete-bottom button {
  padding: 8px 12px;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

/* Отдельные стили для удаления и отмены */
.delete-button {
  background-color: #e53935; /* красный */
  color: #fff;
}

.cancel-button {
  background-color: #757575; /* серый */
  color: #fff;
}

/* Блок кнопки удаления, чтобы отделить её от списка слов */
.mass-delete-bottom {
  margin-top: 20px;
}

/* Список слов */
ul {
  list-style-type: none;
  padding: 0;
}

/* Чтобы слова с чекбоксами были по центру, используем flex + justify-content */
li {
  margin: 10px 0;
  display: flex;
  align-items: center;
  justify-content: center; /* центрируем */
}

/* Чекбокс чуть отступает от текста */
li input[type="checkbox"] {
  margin-right: 10px;
}

/* Верхние углы */
.top-left {
  position: absolute;
  top: 20px;
  left: 20px;
}

.top-right {
  position: absolute;
  top: 20px;
  right: 20px;
}

/* Кнопка загрузки / Upload Image */
button {
  padding: 10px 20px;
  background-color: #42b983;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}
button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* Поле выбора файла */
input[type="file"] {
  display: block;
  margin: 20px auto;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
}
</style>