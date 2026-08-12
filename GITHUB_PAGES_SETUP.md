# GitHub Pages setup — AI4ALL7C

This folder is ready to publish as a static GitHub Pages site.

## Recommended setup for the existing repository

Because the AI4ALL7C repository already contains Python/model files, keep the website in the `docs/` folder rather than replacing the repository root.

1. Copy the generated `docs/` folder into the root of `https://github.com/12chenec/AI4ALL7C`.
2. Commit and push the new files.
3. On GitHub, open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select your default branch (usually `main`) and the **`/docs`** folder, then save.
6. GitHub will publish the site at a URL similar to `https://12chenec.github.io/AI4ALL7C/`.

## Files

```text
docs/
├── index.html       # page structure + project content
├── styles.css       # complete responsive visual design
├── script.js        # mobile nav, reveal animation, interactive metric chart
├── .nojekyll        # tells GitHub Pages to serve files directly
└── assets/
    └── favicon.svg
```

## Editing quick guide

- **Hero text / team / links:** `docs/index.html`
- **Colors:** CSS variables at the top of `docs/styles.css`
- **Model numbers:** `metricData` at the top of `docs/script.js`
- **GitHub URL:** search `https://github.com/12chenec/AI4ALL7C` in `docs/index.html`
- **Streamlit demo:** search `https://ai4all7c-svsehmuzxx4dv5jnwsjzbq.streamlit.app/` in `docs/index.html`

## Notes

- The site has no external JavaScript or CSS dependencies.
- It is responsive and supports reduced-motion preferences.
- The interactive model chart uses the project's held-out test metrics.
- The page explicitly flags the known `admits_per100k` denominator issue rather than presenting that feature as unqualified evidence.
