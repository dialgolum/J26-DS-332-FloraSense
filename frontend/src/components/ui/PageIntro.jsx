export default function PageIntro({ eyebrow, title, description, action }) {
  return <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="mb-2 text-xs font-bold uppercase tracking-[.16em] text-leaf">{eyebrow}</p><h2 className="font-display text-3xl font-extrabold tracking-tight text-forest sm:text-[2.1rem]">{title}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-muted">{description}</p></div>{action}</div>;
}
