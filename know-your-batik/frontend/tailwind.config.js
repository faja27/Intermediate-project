/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'batik-brown': '#3D1A00',
        'batik-gold': '#C8960C',
        'batik-cream': '#FAF3E0',
        'batik-red': '#8B1A1A',
        'batik-teal': '#1A4A4A',
      },
      fontFamily: {
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      backgroundImage: {
        'batik-pattern': `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Cg stroke='%23C8960C' fill='none' opacity='0.12'%3E%3Cellipse cx='20' cy='20' rx='16' ry='10' stroke-width='0.8'/%3E%3Cellipse cx='20' cy='20' rx='10' ry='16' stroke-width='0.8'/%3E%3Cellipse cx='60' cy='20' rx='16' ry='10' stroke-width='0.8'/%3E%3Cellipse cx='60' cy='20' rx='10' ry='16' stroke-width='0.8'/%3E%3Cellipse cx='20' cy='60' rx='16' ry='10' stroke-width='0.8'/%3E%3Cellipse cx='20' cy='60' rx='10' ry='16' stroke-width='0.8'/%3E%3Cellipse cx='60' cy='60' rx='16' ry='10' stroke-width='0.8'/%3E%3Cellipse cx='60' cy='60' rx='10' ry='16' stroke-width='0.8'/%3E%3Ccircle cx='40' cy='40' r='8' stroke-width='0.8'/%3E%3Ccircle cx='20' cy='20' r='2' fill='%23C8960C'/%3E%3Ccircle cx='60' cy='20' r='2' fill='%23C8960C'/%3E%3Ccircle cx='20' cy='60' r='2' fill='%23C8960C'/%3E%3Ccircle cx='60' cy='60' r='2' fill='%23C8960C'/%3E%3C/g%3E%3C/svg%3E")`,
      },
      animation: {
        'spin-slow': 'spin 2s linear infinite',
      },
    },
  },
  plugins: [],
}
