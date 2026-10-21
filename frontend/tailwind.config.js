/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      colors: {
        // Deep-navy canvas + frosted-glass surfaces for the "AI lab" look.
        base: {
          950: '#05070f',
          900: '#0a0e1a',
          850: '#0d1220',
          800: '#111730',
          700: '#1a2140',
        },
        accent: {
          indigo: '#818cf8',
          cyan: '#22d3ee',
          rose: '#fb7185',
          emerald: '#34d399',
          amber: '#fbbf24',
        },
      },
      backgroundImage: {
        'grid-fade':
          'radial-gradient(circle at 50% 0%, rgba(129,140,248,0.12), transparent 60%), radial-gradient(circle at 80% 90%, rgba(34,211,238,0.08), transparent 50%)',
      },
      boxShadow: {
        glass: '0 8px 32px -8px rgba(0,0,0,0.6)',
        glow: '0 0 0 1px rgba(129,140,248,0.15), 0 8px 40px -8px rgba(129,140,248,0.25)',
      },
      keyframes: {
        'fade-in': { '0%': { opacity: 0, transform: 'translateY(6px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out both',
      },
    },
  },
  plugins: [],
}
