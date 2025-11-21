import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'https://style-generator-service-372485790810.asia-south1.run.app',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
