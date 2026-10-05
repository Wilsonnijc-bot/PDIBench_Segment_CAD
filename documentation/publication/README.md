# Current publication

`release.json` explicitly selects the current robot-link runs and the 42-video/420-pair object result. It inventories every local published file with its existing hash. Website files live in `documentation/website/`; image packs remain in `web-assets/` so existing raw-image URLs keep working.

`python documentation/publication/build.py` verifies the bundle. Add `--output NEW_DIRECTORY` to stage an exact copy containing `documentation/website/` and `web-assets/`. This command avoids replaying a base builder that references earlier object cohorts. It is a frozen-release materializer, not a new-run promotion tool.

`builders/` preserves the previous assembly code. Its release historically required the case-update/normalization scripts now under `object/archive/release_updates/`; do not use the old base builder alone to replace the current site. Future publication design should consume one explicit run selection. Existing content-addressed asset names and external release/raw-image URLs remain unchanged.

## Repository rename and republication

The current site is https://wilsonnijc-bot.github.io/robot_object_deformation_detect/, published by `.github/workflows/publish-website.yml` from `documentation/website/` on `main`. The republication updates repository URLs in the site, asset manifests, gallery and latest-results link. Scientific outputs, compressed replay evidence, image packs and historical provenance snapshots remain unchanged. `release.json` records the updated published-file hashes.

## Website source and deployment

GitHub Pages uses Actions to publish `documentation/website/` at the existing site URL. The workflow verifies `release.json` before uploading the website. `web-assets/` remains at its existing path because the gallery reads these image packs directly from GitHub. Top-level `results/` links are local only and are not tracked or pushed.
