# Visual SSL Digest Site

GitHub Pages MVP for the daily general visual self-supervised learning digest.

Live site:

https://jaywalk18.github.io/visual-ssl-digest/

## Current Shape

- `index.html`: front page
- `issues/2026-05-25.html`: latest daily issue page
- `papers/*.html`: paper detail pages
- `pages/catalog.html`: catalog
- `pages/timeline.html`: timeline
- `assets/mineru/<arxiv-id>/`: MinerU `full.md`, content JSON, and extracted figures

## Daily Automation Step

After the existing Markdown/Feishu report finishes:

1. Download selected P1/P2 PDFs to `staging_pdfs/`.
2. Run MinerU batch extraction:

   ```powershell
   python C:\Users\Administrator\.claude\skills\paper-digest-site\scripts\submit_mineru.py `
     --in H:\Desktop\visual_ssl_digest_site\staging_pdfs `
     --out H:\Desktop\visual_ssl_digest_site\assets\mineru
   ```

3. Rebuild the static site:

   ```powershell
   python C:\Users\Administrator\Documents\New project\scripts\build_visual_ssl_site.py
   Copy-Item "C:\Users\Administrator\Documents\New project\scripts\build_visual_ssl_site.py" `
     "H:\Desktop\visual_ssl_digest_site\scripts\build_visual_ssl_site.py" -Force
   ```

4. Commit and push:

   ```powershell
   cd H:\Desktop\visual_ssl_digest_site
   git add -A
   git commit -m "daily: YYYY-MM-DD"
   git push
   ```

## Notes

- The latest issue is 2026-05-25. It is currently a text-first web issue; MinerU figure extraction has not yet been run for the 2026-05-25 papers.
- The site uses MinerU figures for the 2026-05-24/older figure-rich pages: TC-JEPA, ResDreamer, Cambrian-P, DeltaDirect, GeoWeaver, and Pairwise Modalities.
- Current generator is still an MVP: paper metadata is curated in Python. Next step is to load daily paper records from the dedupe JSON and Markdown reports.
- Keep Feishu `latest.md` as the compact text push; use this site as the figure-rich archive.
