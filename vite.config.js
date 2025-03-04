import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/upload': {
        target: 'http://localhost:5137', // Адрес вашего backend'а
        changeOrigin: true,              // Разрешаем изменение заголовков
        rewrite: (path) => path.replace(/^\/upload/, '/upload'), // Перенаправляем на /upload
      },
    },
  },
})
