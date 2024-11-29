/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'selector',
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {},
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: [
      {
        lynxkite: {
          primary: 'oklch(75% 0.2 55)',
          secondary: 'oklch(75% 0.13 230)',
          accent: 'oklch(55% 0.25 320)',
          neutral: 'oklch(35% 0.1 240)',
          'base-100': '#ffffff',
        },
      },
    ],
  },
};
