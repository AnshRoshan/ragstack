# Deploying the website

The site is a single static file (`website/index.html`) — no build step, no
dependencies. Pick any one of these:

## GitHub Pages (recommended, already wired)

A workflow at `.github/workflows/pages.yml` deploys `website/` on every push to
`main`. Enable it once:

```bash
gh api repos/AnshRoshan/ragstack/pages -X POST -f build_type=workflow
```

Site appears at `https://anshroshan.github.io/ragstack/` after the first run.

## Vercel

```bash
cd website
npx vercel --prod
```

Or in the Vercel dashboard: import the repo, set **Root Directory** to
`website`, framework preset **Other**. A `vercel.json` is included.

## Netlify

```bash
cd website
npx netlify deploy --prod --dir .
```

Or drag-and-drop the `website` folder at app.netlify.com/drop. A
`netlify.toml` is included.

## The live-demo section

The "Live demo" section calls `http://127.0.0.1:8000` — the visitor's own
machine. If RAGStack isn't running there, it shows setup instructions instead.
No data ever reaches the hosted site or its author.
