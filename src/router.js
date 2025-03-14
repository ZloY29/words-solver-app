import { createRouter, createWebHistory } from 'vue-router'
import Home from './components/Home.vue'
import DictionaryManager from './components/DictionaryManager.vue'

const routes = [
    { path: '/', name: 'Home', component: Home },
    { path: '/manage', name: 'DictionaryManager', component: DictionaryManager },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
})

export default router