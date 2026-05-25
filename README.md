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

Run the publisher once. It reads `H:\Desktop\visual_ssl_paper_reports\latest.md`, parses the paper index, downloads the selected arXiv PDFs, runs MinerU, rebuilds the site, commits, and pushes.

```powershell
python "C:\Users\Administrator\Documents\New project\scripts\publish_visual_ssl_site.py"
```

Useful flags:

- `--date YYYY-MM-DD`: publish a specific report instead of `latest.md`.
- `--no-mineru`: rebuild the site without re-running PDF extraction.
- `--no-push`: commit locally but skip `git push`.
- `--dry-run`: only parse the report and print selected arXiv IDs.

The expanded manual steps remain:

1. Download selected indexed PDFs to `staging_pdfs/YYYY-MM-DD/`.
2. Run MinerU batch extraction:

   ```powershell
   python C:\Users\Administrator\.claude\skills\paper-digest-site\scripts\submit_mineru.py `
     --in H:\Desktop\visual_ssl_digest_site\staging_pdfs\YYYY-MM-DD `
     --out H:\Desktop\visual_ssl_digest_site\assets\mineru
   ```

3. Rebuild the static site:

   ```powershell
   python C:\Users\Administrator\Documents\New project\scripts\build_visual_ssl_site.py
   Copy-Item "C:\Users\Administrator\Documents\New project\scripts\build_visual_ssl_site.py" `
     "H:\Desktop\visual_ssl_digest_site\scripts\build_visual_ssl_site.py" -Force
   ```

4. Commit and push.

   ```powershell
   cd H:\Desktop\visual_ssl_digest_site
   git add -A
   git commit -m "daily: YYYY-MM-DD"
   git push
   ```

## Notes

- The latest issue is 2026-05-25. MinerU figure extraction has been run for TextTeacher, MDM, EvoVid, Ablate-to-Validate, Neural Collapse by Design, and SEST.
- The front page and issue pages include a floating CCF A/B deadline reminder. Keep CCF C out of the primary reminder unless it is only being logged for context.
- The site uses MinerU figures for the 2026-05-24/older figure-rich pages: TC-JEPA, ResDreamer, Cambrian-P, DeltaDirect, GeoWeaver, and Pairwise Modalities.
- Current generator loads the latest Markdown report's paper index first, then falls back to curated metadata for older/deep-read pages.
- Keep Feishu `latest.md` as the compact text push; use this site as the figure-rich archive.
