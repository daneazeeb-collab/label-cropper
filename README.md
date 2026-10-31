# Label Cropper

Crops eBay shipping-label PDFs to 4 × 6 for a thermal printer and saves each file over itself, keeping its name.

- **Open** — pick labels in a Finder window; each opens in its own tab.
- **Rotate**, **Crop to 4×6**, **Print**, **Save**.

Saving never changes the label itself. Only the page frame and rotation change, written as an update appended to the file. Before anything is written, five checks must pass: original bytes kept, drawing unchanged, nothing cut off, pixel-identical render, and true 4 × 6 size.

## Building

Every push to `main` builds `Label-Cropper.dmg` on a GitHub Mac (see **Actions**). Download it from the run's *Artifacts*.

Run from source: `pip install -r requirements.txt && python label_cropper.py` (add `--browser` to use a browser tab instead of a window).
