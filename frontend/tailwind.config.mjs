/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{ts,tsx,js,jsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:       "#0b0f17",
        panel:    "#111827",
        panel2:   "#1f2937",
        border:   "#1f2937",
        text:     "#e5e7eb",
        muted:    "#94a3b8",
        teal:     "#1D9E75",
        coral:    "#D85A30",
        amber:    "#BA7517",
        purple:   "#7F77DD",
        red:      "#E24B4A",
      },
    },
  },
  plugins: [],
};
