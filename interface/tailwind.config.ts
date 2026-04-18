import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#1a1814", 2: "#3d3a35", 3: "#6b6760" },
        bg: { DEFAULT: "#faf8f4", 2: "#f3f0ea", 3: "#eae6de" },
        rule: "#d8d4cc",
        rule2: "#ece9e3",
        accent: { DEFAULT: "#2d5a3d", 2: "#4a8c62", 3: "#c8e6d0" },
        coral: "#b84a2a",
        amber: "#c18a2a",
      },
      fontFamily: {
        serif: ['"DM Serif Display"', "serif"],
        sans: ['"Instrument Sans"', "sans-serif"],
        mono: ['"DM Mono"', "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
