import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  base: './', // Relative base path enables flawless hosting on GitHub Pages sub-paths
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: true, // Listen on all network interfaces (0.0.0.0)
    allowedHosts: true, // Allow Tailscale and custom domains
  },
})
