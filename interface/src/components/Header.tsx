export function Header() {
  return (
    <header className="relative overflow-hidden bg-ink px-4 pb-12 pt-14 text-white md:px-6">
      <div className="pointer-events-none absolute -right-10 -top-10 h-[300px] w-[300px] rounded-full bg-accent opacity-[0.08]" />
      <div className="pointer-events-none absolute -bottom-20 -left-10 h-[220px] w-[220px] rounded-full bg-coral opacity-[0.06]" />
      <div className="relative z-[1] mx-auto max-w-[780px]">
        <p className="mb-4 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#9fe1cb]">
          Results report &middot; Deep reinforcement learning &middot; Spring 2026
        </p>
        <h1 className="mb-4 max-w-[620px] font-serif text-[clamp(1.65rem,4vw,2.75rem)] font-normal leading-tight">
          Learning when to think:{" "}
          <em className="text-[#9fe1cb] not-italic">
            does the model learn to spend less on easy and more on hard?
          </em>
        </h1>
        <p className="mb-6 max-w-[580px] text-[15px] leading-relaxed text-white/75">
          An RL policy with three actions (continue, refine, terminate) trained on
          Qwen2.5-Math-7B-Instruct with LoRA, GRPO, ALP reward, and DeGRPO weighting.
          Tested on MATH-500.
        </p>
        <div className="mb-5 flex flex-wrap gap-2">
          {["Ivan Andhika · u1590903", "Hao Ren · u1527543", "Ammon Gleason · u1221580"].map(
            (a) => (
              <span
                key={a}
                className="rounded-full border border-white/15 bg-white/[0.08] px-3.5 py-1 text-[12.5px] text-white/85"
              >
                {a}
              </span>
            )
          )}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-white/50">
          <span>University of Utah · CS 6955</span>
          <a
            className="text-white/70 underline decoration-white/25 hover:text-white"
            href="../project_docs/Final proposal - Learning When to Think.pdf"
          >
            Final proposal (PDF)
          </a>
          <a
            className="text-white/70 underline decoration-white/25 hover:text-white"
            href="../docs/results_h1_h3.md"
          >
            Raw result · docs/results_h1_h3.md
          </a>
        </div>
      </div>
    </header>
  );
}
