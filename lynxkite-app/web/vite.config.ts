import react from "@vitejs/plugin-react-swc";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  build: {
    chunkSizeWarningLimit: 3000,
  },
  esbuild: {
    supported: {
      // For dynamic imports.
      "top-level-await": true,
    },
  },
  plugins: [react(), Icons({ compiler: "jsx", jsx: "react" })],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
