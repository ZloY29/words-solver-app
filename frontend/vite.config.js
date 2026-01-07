import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/upload": { target: "http://localhost:8000", changeOrigin: true },
      "/add_word": { target: "http://localhost:8000", changeOrigin: true },
      "/remove_words": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
