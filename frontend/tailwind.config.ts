import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#00E5FF',
        background: '#0F172A',
        surface: '#111827',
        accent: '#38BDF8',
        success: '#22C55E',
        danger: '#EF4444',
      },
    },
  },
  plugins: [],
} satisfies Config;
