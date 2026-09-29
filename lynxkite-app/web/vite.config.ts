import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react-swc";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";

// https://vite.dev/config/
export default defineConfig({
  // Set the base path to be relative. Then in index.html we set it to be absolute.
  // We can then modifying index.html if needed, for example for static exports.
  base: "./",
  build: {
    chunkSizeWarningLimit: 3000,
    sourcemap: true,
  },
  esbuild: {
    supported: {
      // For dynamic imports.
      "top-level-await": true,
    },
  },
  plugins: [react(), Icons({ compiler: "jsx", jsx: "react" }), tailwindcss()],
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
