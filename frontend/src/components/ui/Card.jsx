export default function Card({ children, className = "", ...props }) {
  return <section className={`rounded-2xl border border-line bg-white shadow-soft ${className}`} {...props}>{children}</section>;
}
