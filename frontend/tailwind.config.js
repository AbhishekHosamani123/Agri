/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'chat-dark': '#212121',
        'chat-gray': '#2f2f2f',
        'chat-light': '#343541',
      }
    },
  },
  plugins: [],
}

