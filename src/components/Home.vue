<template>
    <div class="home">
      <div class="top-right">
        <router-link to="/manage">
          <img src="/book.svg" alt="Manage Dictionary" class="icon-button" />
        </router-link>
      </div>
      <h1>Word Finder</h1>
      <input type="file" accept="image/*" @change="handleFileChange" />
      <button @click="uploadImage" :disabled="!selectedImage">Upload Image</button>
      <div v-if="loading">Loading...</div>
      <div v-if="results.length > 0">
        <h2>Found Words:</h2>
        <ul>
          <li v-for="(word, index) in results" :key="index">
            {{ word.name }} - {{ word.score }} points
          </li>
        </ul>
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
    },
  };
  </script>
  
  <style scoped>
  .top-right {
    position: absolute;
    top: 20px;
    right: 20px;
  }
  .icon-button {
    width: 32px;   /* подберите нужный размер */
    height: 32px;
    cursor: pointer;
  }
  .home {
    text-align: center;
    margin-top: 50px;
  }
  input[type="file"] {
    display: block;
    margin: 20px auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
  }
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
  ul {
    list-style-type: none;
    padding: 0;
  }
  li {
    margin: 10px 0;
  }
  </style>