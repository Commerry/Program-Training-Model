import { fileURLToPath, URL } from 'node:url'
import vue from '@vitejs/plugin-vue'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendPort = env.BACKEND_PORT || '64031'
  const frontendPort = Number(env.FRONTEND_PORT || 64030)

  return {
    plugins: [vue()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    server: {
      // Bind every interface so the dev server is reachable by IP from another
      // machine. Set FRONTEND_HOST=127.0.0.1 to keep it on this machine only.
      host: env.FRONTEND_HOST || '0.0.0.0',
      port: frontendPort,
      strictPort: true,
      proxy: {
        // Proxying keeps the API same-origin in development, so the session
        // cookie is sent without any CORS or SameSite configuration.
        '/api': {
          // Always loopback: the browser talks to this dev server, which
          // forwards to the API, so the API itself never needs to be exposed.
          target: `http://127.0.0.1:${backendPort}`,
          changeOrigin: true,
          // Training runs and large uploads far outrun the default timeout.
          timeout: 0,
          proxyTimeout: 0
        }
      }
    },
    build: {
      outDir: 'dist',
      sourcemap: mode !== 'production',
      chunkSizeWarningLimit: 700
    }
  }
})
