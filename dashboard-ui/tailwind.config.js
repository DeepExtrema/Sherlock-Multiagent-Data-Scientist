/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        ink: '#e6faff',
        surface: '#0c1117',
        line: '#121a24',
        accent: '#22d3ee',
      },
    },
  },
  plugins: [],
}


