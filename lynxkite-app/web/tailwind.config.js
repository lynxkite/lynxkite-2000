/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "selector",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {},
  },
  plugins: [require("daisyui"), require("@tailwindcss/typography")],
  daisyui: {
    logs: false,
    themes: [
      {
        lynxkite: {
          primary: "oklch(75% 0.2 55)",
          secondary: "oklch(75% 0.13 230)",
          accent: "oklch(55% 0.25 320)",
          neutral: "oklch(35% 0.1 240)",
          "base-100": "#ffffff",
        },
      },
    ],
  },
};
