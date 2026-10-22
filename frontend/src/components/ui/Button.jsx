export default function Button({ children, variant = "primary", className = "", ...props }) {
  const variants = {
    primary: "bg-forest text-white hover:bg-[#245646] focus-visible:ring-forest",
    secondary: "border border-line bg-white text-forest hover:border-leaf/40 hover:bg-mint/40 focus-visible:ring-leaf",
    quiet: "bg-transparent text-muted hover:bg-mint/60 hover:text-forest focus-visible:ring-leaf",
  };
  return <button className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-xl px-4 text-sm font-bold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-45 ${variants[variant]} ${className}`} {...props}>{children}</button>;
}
