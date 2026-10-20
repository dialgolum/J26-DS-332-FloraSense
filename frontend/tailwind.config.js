/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        forest: "#173f35",
        leaf: "#377a5c",
        mint: "#e8f1e9",
        canvas: "#f6f8f4",
        ink: "#1f302b",
        muted: "#75827c",
        line: "#e5eae4",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        display: ["Manrope", "Inter", "ui-sans-serif", "sans-serif"],
      },
      boxShadow: {
        soft: "0 14px 40px rgba(31, 48, 43, 0.06)",
      },
    },
  },
  plugins: [],
};
