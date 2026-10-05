# PDI selected review

This folder is published by GitHub Pages through `.github/workflows/publish-website.yml` from `main` at `documentation/website/`. The object crop
gallery loads its lossless image packs from `main/web-assets/object-image-packs/`
through GitHub's raw-file URLs. All review files and image packs are maintained
on `main`; separate deployment and image-asset branches are no longer required.

45 original videos grouped with clickable titles for V1 link2/link7, v2_tapip3d link7, and updated-mask link5. Original interactive point-cloud replays are preserved.

The representative mask section contains three Gemini 3.8 Flash persistent SAM prompts with synchronized naive/refined mask comparisons, and three link5 guard examples with exactly two pictures each: initial points and Gemini decisions with the exact points submitted to SAM3. Representative selection, exact coordinates, model identity, and source hashes are recorded in representative/selection.json. Point figures use the same extracted source frame and crop before and after; original source artifacts were not modified.

Replay HTML is stored in lossless gzip; the loader verifies SHA-256 before rendering. Resource URLs resolve through manifest.json. Original videos and prompt images are unchanged release assets. The previous Pages content was replaced at the user's request and remains available in git history.

Current site: https://wilsonnijc-bot.github.io/robot_object_deformation_detect/
