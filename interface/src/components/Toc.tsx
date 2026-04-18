import { useEffect, useState } from "react";
import clsx from "clsx";
import { SECTION_NAV } from "../nav";

export function Toc() {
  const [active, setActive] = useState<string>(SECTION_NAV[0]!.id);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(e.target.id);
        }
      },
      { rootMargin: "-90px 0px -50% 0px", threshold: [0, 0.1, 0.25] }
    );
    for (const { id } of SECTION_NAV) {
      const el = document.getElementById(id);
      if (el) obs.observe(el);
    }
    return () => obs.disconnect();
  }, []);

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <nav className="sticky top-0 z-40 border-b border-rule bg-bg-2/95 backdrop-blur-sm">
      <div className="mx-auto flex max-w-[780px] gap-1 overflow-x-auto px-3 py-2 md:px-4">
        {SECTION_NAV.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            onClick={() => scrollTo(id)}
            className={clsx(
              "shrink-0 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors",
              active === id
                ? "bg-accent text-white"
                : "text-ink-3 hover:bg-accent-3 hover:text-accent"
            )}
          >
            {label}
          </button>
        ))}
      </div>
    </nav>
  );
}
