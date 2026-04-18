# Learning When to Think — results interface

Static React app (Vite + TypeScript + Tailwind + Recharts) that shows the H1–H3 results report in one page: problem, method, setup, charts, and conclusions. Copy is B1 English; numbers match `docs/results_h1_h3.md`.

## Quick start

From the **repo root**:

```bash
cd interface
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

Production build (outputs `interface/dist/`):

```bash
cd interface
npm run build
npm run preview   # optional: serve dist locally
```

Deploy `interface/dist/` to any static host (GitHub Pages, Netlify, etc.). `vite.config.ts` sets `base: './'` so asset paths work when opened from a subfolder.

## Project layout

| Path | Role |
|------|------|
| `src/data/results.ts` | **Single source of truth** for all tables and charts. Update numbers here after a new eval run. |
| `src/sections/*.tsx` | One file per “slide” section (problem, idea, H1, …). |
| `src/components/*.tsx` | Reusable UI (TOC, tables, verdicts). |
| `src/components/charts/*.tsx` | Recharts wrappers (Pareto scatter, bar charts). |
| `src/nav.ts` | Section IDs for the sticky TOC and scroll-spy. |

## Updating results

1. Refresh `docs/results_h1_h3.md` (or your `comparison.json` / CSV pipeline).
2. Edit **`src/data/results.ts`**: `h1Methods`, `h2WithRefine`, `h2NoRefine`, `h3WithRefine`, `h3NoRefine`, `spearman`.
3. If you add/remove sections, update **`src/nav.ts`** and **`src/App.tsx`**.
4. Run `npm run build` to confirm TypeScript and the bundle succeed.

## Links in the UI

Header and footer use relative links (`../docs/...`, `../project_docs/Final proposal - Learning When to Think.pdf`) so they resolve when the app is served from `interface/` during dev, or adjust paths for your deploy URL.

## Stack

- Vite 5, React 18, TypeScript 5
- Tailwind CSS 3 (palette aligned with `proposal.html`)
- Recharts 2
