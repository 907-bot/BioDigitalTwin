/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{ts,tsx,js,jsx,mdx}"],
  theme: {
    extend: {
      colors: {
        bg:        "#070c14",
        "bg-2":    "#0d1422",
        panel:     "#0f1929",
        "panel-2": "#152035",
        "panel-3": "#1a2840",
        border:    "#1e2d42",
        "border-2":"#253548",
        text:      "#e2eaf5",
        muted:     "#7a93b0",
        "muted-2": "#4d6580",
        teal:      "#00d4a0",
        "teal-2":  "#00b887",
        cyan:      "#00bcd4",
        coral:     "#ff6b5b",
        amber:     "#ffa94d",
        purple:    "#a78bfa",
        "purple-2":"#7c6df0",
        red:       "#f87171",
        green:     "#4ade80",
        blue:      "#60a5fa",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "pulse-slow":   "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "glow":         "glow 2s ease-in-out infinite alternate",
        "float":        "float 6s ease-in-out infinite",
        "slide-in":     "slideIn 0.3s ease-out",
        "fade-in":      "fadeIn 0.4s ease-out",
        "scan":         "scan 4s linear infinite",
        "shimmer":      "shimmer 2s infinite",
      },
      keyframes: {
        glow: {
          "0%":   { boxShadow: "0 0 5px rgba(0, 212, 160, 0.2)" },
          "100%": { boxShadow: "0 0 20px rgba(0, 212, 160, 0.5)" },
        },
        float: {
          "0%,100%": { transform: "translateY(0)" },
          "50%":     { transform: "translateY(-6px)" },
        },
        slideIn: {
          "0%":   { transform: "translateX(-10px)", opacity: "0" },
          "100%": { transform: "translateX(0)",     opacity: "1" },
        },
        fadeIn: {
          "0%":   { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        scan: {
          "0%":   { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      backgroundImage: {
        "grid-pattern": "radial-gradient(circle, #1e2d42 1px, transparent 1px)",
        "hero-glow":    "radial-gradient(ellipse 80% 50% at 50% 0%, rgba(0,212,160,0.15) 0%, transparent 60%)",
        "card-glow":    "linear-gradient(135deg, rgba(0,212,160,0.05) 0%, transparent 100%)",
      },
    },
  },
  plugins: [],
};
