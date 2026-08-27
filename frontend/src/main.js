import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import { installAuthInterceptor } from './stores/authStore'
import './assets/styles/fonts.css'
import './assets/styles/main.css'

const app = createApp(App)

app.use(createPinia())
// The interceptor needs the router, and the router guard needs the store, so
// the store must be installed before the router is mounted.
installAuthInterceptor(router)
app.use(router)

app.mount('#app')
