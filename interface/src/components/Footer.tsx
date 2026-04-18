export function Footer() {
  return (
    <footer className="border-t border-rule bg-bg-2 px-4 py-8 md:px-6">
      <div className="mx-auto max-w-[780px] text-[12.5px] leading-relaxed text-ink-3">
        <p>
          Source data:{" "}
          <a className="text-accent hover:underline" href="../docs/results_h1_h3.md">
            docs/results_h1_h3.md
          </a>
          . Code:{" "}
          <a
            className="text-accent hover:underline"
            href="../scripts/analyze_h1_pareto.py"
          >
            analyze_h1_pareto.py
          </a>
          ,{" "}
          <a className="text-accent hover:underline" href="../scripts/analyze_h2_h3.py">
            analyze_h2_h3.py
          </a>
          . CHPC eval logs under <code className="rounded bg-bg-3 px-1 font-mono text-xs">chpc/results/eval/</code>.
        </p>
      </div>
    </footer>
  );
}
