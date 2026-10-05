"""Write the small Markdown-and-images interface for an exact VLM2 request."""
import hashlib
from pathlib import Path


def _save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_interface(folder, *, system_prompt, user_prompt, images, role, model, backend):
    """Replace ``folder`` with README.md and the four current request images."""
    folder = Path(folder)
    image_dir = folder / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    names = ("target_frame.png", "reference_1.png", "reference_2.png", "reference_3.png")
    labels = (
        "Image 1 — selected SAM reseeding frame",
        "Image 2 — annotated reference 1",
        "Image 3 — annotated reference 2",
        "Image 4 — annotated reference 3",
    )
    hashes = []
    for image, name in zip(images, names, strict=True):
        hashes.append(_save_png(image, image_dir / name))
    image_sections = "\n\n".join(
        f"### {label}\n\n![{label}](images/{name})\n\n`sha256: {digest}`"
        for label, name, digest in zip(labels, names, hashes, strict=True)
    )
    markdown = f"""# Current VLM2 request

This folder is the human-readable interface for the latest VLM2 call.

- **Call role:** `{role}`
- **Backend:** `{backend}`
- **Model:** `{model}`
- **Image order:** target frame first, then the three annotated references

## System prompt

```text
{system_prompt}
```

## User task

```text
{user_prompt}
```

## Images sent to VLM2

{image_sections}
"""
    temporary = folder / "README.tmp.md"
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(folder / "README.md")
    return hashes
