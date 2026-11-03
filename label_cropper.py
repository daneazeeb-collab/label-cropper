#!/usr/bin/env python3
"""
Label Cropper — trims eBay shipping-label PDFs to 4x6 and saves each one over itself.

How it changes a file
  The label's drawing is never touched. The tool only changes the page's frame
  (MediaBox / CropBox) and its /Rotate value, and it writes those changes as a
  PDF "incremental update" appended to the end of the file. Every original byte
  of the file stays exactly where it was.

Opening files
  Nothing is scanned. Click Open to choose labels in a Finder window; only the
  files you choose there can be read or saved.

Checks run before a save is accepted (any failure = nothing is written)
  1. Original bytes intact: the new file begins with the old file, byte for byte.
  2. Drawing unchanged: every page's content streams and every image/form it
     uses decode to identical bytes before and after.
  3. Nothing cut off: every non-white pixel on the whole original page lies
     inside the new crop window.
  4. Same picture: the saved page renders pixel-identical to the original page
     viewed through the same window, and no ink touches the edge of the label.
  5. Actual size: the label is not scaled, so barcodes print at their original size.
"""
import hashlib
import time
_T0 = time.monotonic()                      # for the startup timings in the log

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import importlib
import logging


# The PDF libraries take most of the startup time (numpy alone ~0.4 s, more under Rosetta), so
# they load in the background while the window opens. Code below uses them as if imported.
class _Lazy:
    def __init__(self, module, attr=None):
        self._module, self._attr, self._obj = module, attr, None

    def _load(self):
        if self._obj is None:
            m = importlib.import_module(self._module)
            self._obj = getattr(m, self._attr) if self._attr else m
            if self._module.startswith("pypdf"):
                logging.getLogger("pypdf").setLevel(logging.ERROR)
        return self._obj

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __call__(self, *a, **k):
        return self._load()(*a, **k)


np = _Lazy("numpy")
pdfium = _Lazy("pypdfium2")
PdfReader = _Lazy("pypdf", "PdfReader")
PdfWriter = _Lazy("pypdf", "PdfWriter")
NameObject = _Lazy("pypdf.generic", "NameObject")
NumberObject = _Lazy("pypdf.generic", "NumberObject")
RectangleObject = _Lazy("pypdf.generic", "RectangleObject")


def warm_up():
    """Load the PDF libraries and draw one tiny page, so the first real label is instant."""
    t = time.monotonic()
    for lazy in (np, pdfium, PdfReader, NameObject):
        lazy._load()
    try:
        _render_displayed(_synthetic_label(), 0, 0.1)
    except Exception:
        pass
    startup_log(f"PDF libraries ready in {time.monotonic() - t:.2f}s")


def startup_log(msg):
    print(f"[{time.monotonic() - _T0:6.2f}s] {msg}", flush=True)

# PDFium is not thread-safe, and the window's server answers requests on several threads at
# once (previews, saves, print checks). Every use of PDFium goes through this one lock;
# without it PDFium corrupts its own memory and the app crashes.
PDFIUM_LOCK = threading.RLock()


def _pdfium_serialized(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        with PDFIUM_LOCK:
            return fn(*a, **k)
    return wrapper


PT = 72.0
STOCK_W, STOCK_H = 4 * PT, 6 * PT          # finished label, portrait
DETECT_SCALE = 200 / PT                     # 200 dpi for finding ink
INK = 250                                   # any channel below this counts as ink
PAD = 3.0                                   # points of air around detected ink
LABEL_WORDS = re.compile(r"TRACKING|USPS|POSTAGE|UPS\b|FEDEX|SHIP TO", re.I)


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def _ink_bbox(arr):
    mask = (arr[..., :3] < INK).any(axis=2)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


_INK_CACHE = {}


def ink_bounds(pdf_bytes, page_index=0):
    """Cached: opening a tab, cropping and saving all ask the same question of the same bytes."""
    import hashlib
    key = (hashlib.sha1(pdf_bytes).digest(), page_index)
    hit = _INK_CACHE.get(key)
    if hit is None:
        hit = _ink_bounds(pdf_bytes, page_index)
        if len(_INK_CACHE) > 64:
            _INK_CACHE.clear()
        _INK_CACHE[key] = hit
    return hit


@_pdfium_serialized
def _ink_bounds(pdf_bytes, page_index=0):
    """Bounds of all ink on the whole MediaBox, in PDF user-space points (l, b, r, t)."""
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        page = doc[page_index]
        l, b, r, t = page.get_mediabox()
        page.set_cropbox(l, b, r, t)
        page.set_rotation(0)
        arr = page.render(scale=DETECT_SCALE).to_numpy()
        box = _ink_bbox(arr)
        if box is None:
            return None, (l, b, r, t)
        x0, y0, x1, y1 = box
        s = DETECT_SCALE
        # pixel edges -> points, rounded outward by one pixel
        return (l + (x0 - 1) / s, t - (y1 + 2) / s, l + (x1 + 2) / s, t - (y0 - 1) / s), (l, b, r, t)
    finally:
        doc.close()


def page_state(pdf_bytes, page_index=0):
    r = PdfReader(io.BytesIO(pdf_bytes))
    p = r.pages[page_index]
    rot = int(p.get("/Rotate", 0) or 0) % 360
    return rot, [float(v) for v in p.cropbox], [float(v) for v in p.mediabox], len(r.pages)


def make_plan(pdf_bytes, page_index=0, rotation=None, crop=True):
    """Decide the crop window and rotation. rotation=None means pick automatically."""
    ink, media = ink_bounds(pdf_bytes, page_index)
    cur_rot, cur_crop, _, _ = page_state(pdf_bytes, page_index)
    if ink is None:
        raise ValueError("This page is blank.")
    il, ib, ir, it = ink[0] - PAD, ink[1] - PAD, ink[2] + PAD, ink[3] + PAD
    w, h = ir - il, it - ib
    landscape = w > h
    if rotation is None:
        sideways = cur_rot % 180 == 90
        if crop and landscape != sideways:
            # eBay draws the label sideways with the postage block on the right.
            rotation = 270 if landscape else 0
        else:
            rotation = cur_rot
    rotation %= 360
    ink = [float(v) for v in ink]
    plan = {"rotation": rotation, "crop": crop, "ink": ink, "oversize": False}
    if not crop:
        plan["window"] = cur_crop
        plan["media"] = [float(v) for v in media]
        return plan
    sideways = rotation % 180 == 90
    W, H = (STOCK_H, STOCK_W) if sideways else (STOCK_W, STOCK_H)
    if w <= W and h <= H:
        cx, cy = (il + ir) / 2, (ib + it) / 2
        win = [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2]
    else:
        win = [il, ib, ir, it]
        plan["oversize"] = True
    ml, mb, mr, mt = media
    win = [round(float(v), 3) for v in win]
    plan["window"] = win
    plan["media"] = [min(ml, win[0]), min(mb, win[1]), max(mr, win[2]), max(mt, win[3])]
    return plan


def apply_plan(pdf_bytes, plan, page_index=0):
    """Return new file bytes: the original bytes plus an appended update that only changes the frame."""
    w = PdfWriter(io.BytesIO(pdf_bytes), incremental=True)
    page = w.pages[page_index]
    page[NameObject("/MediaBox")] = RectangleObject(plan["media"])
    page[NameObject("/CropBox")] = RectangleObject(plan["window"])
    page[NameObject("/Rotate")] = NumberObject(plan["rotation"])
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


# ----------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------
def _stream_fingerprints(pdf_bytes):
    """Hash every page's decoded content plus every XObject (images, forms) it draws."""
    r = PdfReader(io.BytesIO(pdf_bytes))
    prints = []
    for p in r.pages:
        h = hashlib.sha256()
        c = p.get_contents()
        h.update(c.get_data() if c is not None else b"")
        seen = set()

        def walk(res):
            res = res.get_object() if res is not None else None
            if not res or "/XObject" not in res:
                return
            xo = res["/XObject"].get_object()
            for name in sorted(xo.keys()):
                ref = xo.raw_get(name)
                key = getattr(ref, "idnum", None) or id(ref)
                if key in seen:
                    continue
                seen.add(key)
                obj = ref.get_object()
                h.update(name.encode())
                try:
                    h.update(obj.get_data())
                except Exception:
                    h.update(obj._data if hasattr(obj, "_data") else b"")
                if obj.get("/Subtype") == "/Form":
                    walk(obj.get("/Resources"))

        walk(p.get("/Resources"))
        prints.append(h.hexdigest())
    return prints


@_pdfium_serialized
def _render_displayed(pdf_bytes, page_index, scale, window=None, rotation=None):
    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        page = doc[page_index]
        if window is not None:
            page.set_mediabox(*window)
            page.set_cropbox(*window)
        if rotation is not None:
            page.set_rotation(rotation)
        return page.render(scale=scale).to_numpy()[..., :3].copy()
    finally:
        doc.close()


def print_copy(pdf_bytes, page_index=0):
    """A one-page PDF for the printer: exactly the label, upright, with the page starting at 0,0.

    Printers differ in whether they honour a PDF's crop window and rotation. Some use the full
    letter page instead and print a blank corner of it. This copy has no window and no rotation
    left to interpret: the page *is* the label. The drawing is wrapped, not changed.
    """
    from pypdf import Transformation
    w = PdfWriter()
    w.append(PdfReader(io.BytesIO(pdf_bytes)), pages=[page_index])
    page = w.pages[0]
    cb = page.cropbox
    l, b, r, t = float(cb.left), float(cb.bottom), float(cb.right), float(cb.top)
    rot = int(page.get("/Rotate", 0) or 0) % 360
    W, H = r - l, t - b
    # move the window to the origin, then turn it clockwise by `rot` the way a viewer would
    tr = Transformation().translate(-l, -b)
    if rot == 90:
        tr = tr.rotate(-90).translate(0, W)
        W, H = H, W
    elif rot == 180:
        tr = tr.rotate(180).translate(W, H)
    elif rot == 270:
        tr = tr.rotate(90).translate(H, 0)
        W, H = H, W
    page.add_transformation(tr)
    box = RectangleObject([0, 0, W, H])
    for name in ("/MediaBox", "/CropBox", "/TrimBox", "/BleedBox", "/ArtBox"):
        if name in page or name in ("/MediaBox", "/CropBox"):
            page[NameObject(name)] = box
    page[NameObject("/Rotate")] = NumberObject(0)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


def check_print_copy(source, copy, page_index=0):
    """The printer copy must look exactly like the label on screen, and must not be blank."""
    s = 150 / PT
    want = _render_displayed(source, page_index, s)
    got = _render_displayed(copy, 0, s)
    r = PdfReader(io.BytesIO(copy)).pages[0]
    mb = [float(v) for v in r.mediabox]
    framed = mb[0] == 0 and mb[1] == 0 and int(r.get("/Rotate", 0) or 0) == 0
    if _ink_bbox(got) is None:
        return False, "The page for the printer came out blank."
    if not framed:
        return False, "The page for the printer isn't framed correctly."
    if abs(got.shape[0] - want.shape[0]) > 2 or abs(got.shape[1] - want.shape[1]) > 2:
        return False, "The page for the printer is a different size from the label."
    # The two are drawn by different routes (rotated view vs. rotated page), so edges can land a
    # pixel apart. Compare at a third of the resolution: that absorbs rounding but still catches
    # a label that is shifted, turned, cut off or missing.
    from PIL import Image
    n = 200
    m = max(1, round(n * want.shape[0] / want.shape[1]))
    A = np.asarray(Image.fromarray(want).convert("L").resize((n, m), Image.BOX), float)
    B = np.asarray(Image.fromarray(got).convert("L").resize((n, m), Image.BOX), float)
    d = np.abs(A - B)
    if d.mean() > 10 or d.max() > 140:
        return False, "The page for the printer doesn't match the label on screen."
    return True, f"Sending a {mb[2]/PT:.2f} × {mb[3]/PT:.2f} in page that matches the preview."


def verify(orig, new, plan, page_index=0):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("Original bytes kept", new[: len(orig)] == orig and len(new) >= len(orig),
        f"The first {len(orig):,} bytes of the file are unchanged; the update is appended after them.")

    a, b = _stream_fingerprints(orig), _stream_fingerprints(new)
    add("Label drawing unchanged", a == b,
        "Text, barcodes and images decode identically before and after." if a == b
        else "The page drawing differs.")

    ink = plan["ink"]
    win = plan["window"]
    inside = ink[0] >= win[0] and ink[1] >= win[1] and ink[2] <= win[2] and ink[3] <= win[3]
    add("Nothing cut off", inside,
        "Every mark on the original page sits inside the new label edges." if inside
        else "Part of the label would fall outside the crop.")

    s = 150 / PT
    got = _render_displayed(new, page_index, s)
    ref = _render_displayed(orig, page_index, s, window=win if plan["crop"] else None,
                            rotation=plan["rotation"])
    same = got.shape == ref.shape and np.array_equal(got, ref)
    box = _ink_bbox(got)
    clear = box is not None and box[0] > 0 and box[1] > 0 and box[2] < got.shape[1] - 1 and box[3] < got.shape[0] - 1
    add("Same picture", same and (clear or not plan["crop"]),
        "The saved label renders pixel-for-pixel the same as the original, with clear space on all four sides."
        if same and clear else
        "The saved label renders pixel-for-pixel the same as the original." if same else
        "The rendered label does not match the original.")

    r = PdfReader(io.BytesIO(new)).pages[page_index]
    cb = [float(v) for v in r.cropbox]
    wpt, hpt = cb[2] - cb[0], cb[3] - cb[1]
    if plan["rotation"] % 180 == 90:
        wpt, hpt = hpt, wpt
    four_six = abs(wpt - STOCK_W) < 0.5 and abs(hpt - STOCK_H) < 0.5
    if plan["crop"]:
        add("Actual size", True,
            "Page is exactly 4 × 6 in. Print at 100% and barcodes keep their original size." if four_six
            else f"Label is {wpt/PT:.2f} × {hpt/PT:.2f} in, larger than 4 × 6, so printing scales it to fit.")
    return all(c["ok"] for c in checks), checks


def status_of(pdf_bytes, page_index=0):
    """'cropped' when the page already shows as an upright 4x6 holding all the ink."""
    try:
        rot, crop, _, _ = page_state(pdf_bytes, page_index)
        if ink_bounds(pdf_bytes, page_index)[0] is None:
            return "blank"
        w, h = crop[2] - crop[0], crop[3] - crop[1]
        if rot % 180 == 90:
            w, h = h, w
        if abs(w - STOCK_W) > 1 or abs(h - STOCK_H) > 1:
            return "needs crop"
        ink, _ = ink_bounds(pdf_bytes, page_index)
        if ink and ink[0] >= crop[0] - 0.5 and ink[1] >= crop[1] - 0.5 and ink[2] <= crop[2] + 0.5 and ink[3] <= crop[3] + 0.5:
            return "cropped"
        return "needs crop"
    except Exception:
        return "unreadable"


def save_in_place(path, plan, page_index=0):
    """Append the update to the file itself so its identity and creation date are kept."""
    path = Path(path)
    orig = path.read_bytes()
    rot, crop, media, _ = page_state(orig, page_index)
    if rot == plan["rotation"] and all(abs(a - b) < 0.01 for a, b in zip(crop, plan["window"])) \
            and all(abs(a - b) < 0.01 for a, b in zip(media, plan["media"])):
        return True, [{"name": "Already saved", "ok": True,
                       "detail": "This file already matches. Nothing needed to be written."}]
    new = apply_plan(orig, plan, page_index)
    ok, checks = verify(orig, new, plan, page_index)
    if not ok:
        return False, checks
    tail = new[len(orig):]
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() != len(orig):
            return False, checks + [{"name": "File unchanged on disk", "ok": False,
                                      "detail": "The file changed while saving. Nothing was written."}]
        f.write(tail)
        f.flush()
        os.fsync(f.fileno())
    written = path.read_bytes()
    if written != new:
        with open(path, "r+b") as f:
            f.truncate(len(orig))
        return False, checks + [{"name": "Written correctly", "ok": False,
                                  "detail": "The file did not read back correctly, so it was restored."}]
    return True, checks


# ----------------------------------------------------------------------------
# opening files (Finder window)
# ----------------------------------------------------------------------------
CHOOSE_SCRIPT = """
activate
try
    set picked to choose file with prompt "Choose labels to open" of type {"com.adobe.pdf"} default location (path to documents folder) with multiple selections allowed
on error number -128
    return ""
end try
set out to ""
repeat with f in picked
    set out to out & POSIX path of f & linefeed
end repeat
return out
"""


WINDOW = None                                        # set when running in the app window


def choose_files():
    """Show a Finder open window. Returns the chosen PDF paths ([] if cancelled)."""
    test = os.environ.get("LABEL_TEST_PICK")          # used for automated testing only
    if test is not None:
        return [p for p in test.split(os.pathsep) if p]
    if WINDOW is not None:
        import webview
        picked = WINDOW.create_file_dialog(
            webview.FileDialog.OPEN if hasattr(webview, "FileDialog") else webview.OPEN_DIALOG,
            directory=str(Path.home() / "Documents"), allow_multiple=True,
            file_types=("PDF files (*.pdf)",))
        return list(picked or [])
    r = subprocess.run(["osascript", "-e", CHOOSE_SCRIPT], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "The Finder window couldn't be opened.")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]

# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------
# Talks to the Mac's print system (CUPS) over IPP, the same protocol System Settings uses,
# so the app can see whether a printer is connected and whether a job really finished.
import http.client
import socket
import struct

CUPS_SOCKET = "/private/var/run/cupsd"


class _UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout=5):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def _cups_connections():
    host = os.environ.get("LABEL_CUPS_HOST")          # used for automated testing only
    if host:
        h, _, p = host.partition(":")
        return [lambda: http.client.HTTPConnection(h, int(p or 631), timeout=5)]
    out = []
    if os.path.exists(CUPS_SOCKET):
        out.append(lambda: _UnixHTTP(CUPS_SOCKET))
    out.append(lambda: http.client.HTTPConnection("localhost", 631, timeout=5))
    return out


def _ipp_attr(tag, name, values):
    out = b""
    for i, v in enumerate(values):
        if isinstance(v, int):
            v = struct.pack(">i", v)
        elif isinstance(v, str):
            v = v.encode()
        n = name.encode() if i == 0 else b""
        out += struct.pack(">BH", tag, len(n)) + n + struct.pack(">H", len(v)) + v
    return out


def _ipp_parse(data):
    status = struct.unpack(">H", data[2:4])[0]
    i, groups, cur, last = 8, [], None, None
    while i < len(data):
        tag = data[i]; i += 1
        if tag == 0x03:
            break
        if tag < 0x10:
            cur = {}; groups.append((tag, cur)); last = None
            continue
        nl = struct.unpack(">H", data[i:i + 2])[0]; i += 2
        name = data[i:i + nl].decode("utf-8", "replace"); i += nl
        vl = struct.unpack(">H", data[i:i + 2])[0]; i += 2
        raw = data[i:i + vl]; i += vl
        if tag in (0x21, 0x23) and vl == 4:
            val = struct.unpack(">i", raw)[0]
        elif tag == 0x22:
            val = raw != b"\x00"
        else:
            val = raw.decode("utf-8", "replace")
        if name:
            last = name
            cur.setdefault(name, []).append(val)
        elif last and cur is not None:
            cur[last].append(val)
    return status, groups


def ipp(op, path, uri_name, uri, extra=b"", want=()):
    body = struct.pack(">BBHI", 2, 0, op, 1) + b"\x01"
    body += _ipp_attr(0x47, "attributes-charset", ["utf-8"])
    body += _ipp_attr(0x48, "attributes-natural-language", ["en"])
    body += _ipp_attr(0x45, uri_name, [uri])
    body += _ipp_attr(0x42, "requesting-user-name", [os.environ.get("USER", "user")])
    if want:
        body += _ipp_attr(0x44, "requested-attributes", list(want))
    body += extra + b"\x03"
    err = None
    for make in _cups_connections():
        try:
            c = make()
            c.request("POST", path, body, {"Content-Type": "application/ipp"})
            r = c.getresponse()
            data = r.read()
            c.close()
            if r.status != 200:
                raise RuntimeError(f"print system answered HTTP {r.status}")
            status, groups = _ipp_parse(data)
            merged = {}
            for tag, g in groups:
                if tag != 0x01:
                    merged.update(g)
            return status, merged
        except Exception as e:
            err = e
    raise RuntimeError(f"Couldn't reach the Mac's print system ({err}).")


PRINTER_STATES = {3: "idle", 4: "printing", 5: "stopped"}
JOB_STATES = {3: "pending", 4: "held", 5: "printing", 6: "stopped", 7: "canceled", 8: "aborted", 9: "completed"}

# printer-state-reasons -> what to tell you. Checked in this order; first match wins.
REASONS = [
    ("paused", "The printer is paused in macOS.", "resume"),
    ("offline", "The printer isn't connected or is turned off.", None),
    ("connecting-to-device", "The Mac can't find the printer. Check that it's plugged in and turned on.", None),
    ("timed-out", "The printer stopped responding.", None),
    ("shutdown", "The printer is turned off.", None),
    ("media-empty", "The printer is out of labels.", None),
    ("media-needed", "The printer needs labels loaded.", None),
    ("media-jam", "Labels are jammed in the printer.", None),
    ("cover-open", "The printer's cover is open.", None),
    ("door-open", "The printer's door is open.", None),
    ("marker-supply-empty", "The printer is out of ribbon or ink.", None),
    ("cups-missing-filter", "macOS is missing the driver this printer needs.", None),
    ("cups-insecure-filter", "macOS blocked this printer's driver.", None),
]


def _base_reason(r):
    for suf in ("-error", "-warning", "-report"):
        if r.endswith(suf):
            return r[: -len(suf)], suf[1:]
    return r, "error"


def diagnose(state, accepting, reasons, message):
    """Return (problem text or None, fix) for a printer's current condition."""
    reasons = [r for r in reasons if r and r != "none"]
    for key, text, fix in REASONS:
        for r in reasons:
            base, sev = _base_reason(r)
            if base == key and (sev != "warning" or key in ("offline", "paused")):
                return text + (f" ({message})" if message and message.lower() not in text.lower() else ""), fix
    if state == "stopped":
        return "The printer is stopped." + (f" ({message})" if message else ""), "resume"
    if accepting is False:
        return "The printer isn't accepting new jobs right now.", "resume"
    for r in reasons:
        base, sev = _base_reason(r)
        if sev == "error":
            return f"The printer reported a problem: {message or base.replace('-', ' ')}.", None
    return None, None


def printer_status(name):
    status, a = ipp(0x000B, f"/printers/{name}", "printer-uri", f"ipp://localhost/printers/{name}",
                    want=("printer-state", "printer-state-reasons", "printer-state-message",
                          "printer-is-accepting-jobs", "printer-info", "device-uri",
                          "media-supported", "media-default"))
    if status >= 0x0400:
        raise RuntimeError("macOS doesn't know a printer by that name any more.")
    state = PRINTER_STATES.get((a.get("printer-state") or [0])[0], "unknown")
    accepting = (a.get("printer-is-accepting-jobs") or [True])[0]
    reasons = a.get("printer-state-reasons") or []
    message = ((a.get("printer-state-message") or [""])[0] or "").strip()
    problem, fix = diagnose(state, accepting, reasons, message)
    return {"name": name, "label": ((a.get("printer-info") or [""])[0] or name.replace("_", " ")),
            "state": state, "accepting": accepting, "reasons": reasons, "message": message,
            "problem": problem, "fix": fix,
            "media": a.get("media-supported") or [], "media_default": (a.get("media-default") or [""])[0]}


_MEDIA_SIZE = re.compile(r"_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(in|mm)$")


def media_size(name):
    """Size in inches from a PWG media name like 'na_index-4x6_4x6in', or None."""
    m = _MEDIA_SIZE.search(name or "")
    if not m:
        return None
    w, h = float(m.group(1)), float(m.group(2))
    if m.group(3) == "mm":
        w, h = w / 25.4, h / 25.4
    return w, h


def pick_media(status, w_in, h_in):
    """The printer's own name for paper of this size, if it has one."""
    def fits(name):
        sz = media_size(name)
        return sz and sorted((round(sz[0], 1), round(sz[1], 1))) == sorted((round(w_in, 1), round(h_in, 1)))
    if fits(status.get("media_default")):
        return status["media_default"]
    for m in status.get("media") or []:
        if fits(m):
            return m
    return None


def printers():
    names, default = [], None
    try:
        out = subprocess.run(["lpstat", "-a"], capture_output=True, text=True, timeout=5).stdout
        names = [ln.split()[0] for ln in out.splitlines() if ln.strip()]
        d = subprocess.run(["lpstat", "-d"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r":\s*(\S+)", d)
        default = m.group(1) if m else None
    except Exception:
        pass
    items = []
    for n in names:
        try:
            items.append(printer_status(n))
        except Exception as e:
            items.append({"name": n, "label": n.replace("_", " "), "problem": None, "unknown": str(e)})
    return {"printers": items, "default": default if default in names else (names[0] if names else None)}


def lpstat_status(name):
    """Rougher check used only if the print system's status can't be read directly."""
    try:
        out = subprocess.run(["lpstat", "-p", name], capture_output=True, text=True, timeout=5)
    except Exception:
        return {"problem": None, "fix": None}
    text = (out.stdout + out.stderr).lower()
    if out.returncode != 0 or "invalid destination" in text or "does not exist" in text:
        return {"problem": "macOS doesn't know a printer by that name any more.", "fix": None}
    if "disabled" in text:
        return {"problem": "The printer is paused in macOS.", "fix": "resume"}
    return {"problem": None, "fix": None}


def submit(pdf_bytes, printer, title, oversize, media=None):
    """Hand the label to macOS. Returns the job's request id, e.g. 'Rollo-42'."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(pdf_bytes)
        tmp = tf.name
    cmd = ["lp", "-d", printer, "-t", title]
    cmd += ["-o", "fit-to-page"] if oversize else ["-o", "scaling=100"]
    if media:
        cmd += ["-o", f"media={media}"]
    cmd.append(tmp)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            if "does not exist" in out or "nvalid destination" in out:
                raise RuntimeError("macOS doesn't know a printer by that name any more. Nothing was sent.")
            raise RuntimeError((out or "macOS refused the print job.") + " Nothing was sent.")
        m = re.search(r"request id is (\S+)", out)
        if not m:
            raise RuntimeError(f"macOS didn't confirm the print job. It said: {out or 'nothing'}")
        return m.group(1)
    finally:
        threading.Timer(300, lambda: os.path.exists(tmp) and os.remove(tmp)).start()


def job_status(request_id, printer):
    """Where a job is now: printing / done / failed / stuck, with a plain explanation."""
    job_id = int(request_id.rsplit("-", 1)[-1])
    status, a = ipp(0x0009, "/jobs", "job-uri", f"ipp://localhost/jobs/{job_id}",
                    want=("job-state", "job-state-reasons", "job-printer-state-message",
                          "job-state-message", "job-impressions-completed"))
    if status >= 0x0400:
        return {"phase": "failed", "text": "macOS lost track of this print job."}
    js = JOB_STATES.get((a.get("job-state") or [0])[0], "unknown")
    reasons = a.get("job-state-reasons") or []
    msg = ((a.get("job-printer-state-message") or a.get("job-state-message") or [""])[0] or "").strip()
    if js == "completed":
        return {"phase": "done", "text": "The printer received the whole label."}
    if js in ("canceled", "aborted"):
        why = "It was canceled." if js == "canceled" else "macOS gave up on it."
        return {"phase": "failed", "text": f"The label did not print. {why}" + (f" ({msg})" if msg else "")}
    try:
        p = printer_status(printer)
    except Exception:
        p = {"problem": None, "fix": None}
    if js in ("stopped", "held") or p.get("problem"):
        text = p.get("problem") or ("The job is on hold." if js == "held" else "The job stopped.")
        return {"phase": "stuck", "text": text + " The label hasn't printed yet.", "fix": p.get("fix")}
    return {"phase": "printing" if js == "printing" else "waiting", "text": msg or "Sending to the printer…"}


def cancel_job(request_id):
    r = subprocess.run(["cancel", request_id], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "macOS couldn't cancel the job.")


def resume_printer(name):
    errs = []
    for cmd in (["cupsenable", name], ["cupsaccept", name]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            errs.append((r.stderr or r.stdout).strip())
    if errs:
        raise RuntimeError("macOS didn't let the app resume the printer. Open System Settings → "
                           "Printers & Scanners, pick the printer, and click Resume.")


# ----------------------------------------------------------------------------
# app server
# ----------------------------------------------------------------------------
HERE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def _quit_app():
    """Close the window and end the process (used to let an update swap the app)."""
    try:
        if WINDOW is not None:
            WINDOW.destroy()
    finally:
        threading.Timer(1.5, lambda: os._exit(0)).start()


class App:
    def __init__(self):
        import updater
        self.updater = updater.Updater()
        self.update_info = None
        self.opened = set()           # only files you picked in the Finder window can be read or saved
        self.choosing = threading.Lock()

    def allowed(self, path):
        p = Path(path).resolve()
        return str(p) in self.opened and p.suffix.lower() == ".pdf" and p.is_file()


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _plan_from(self, d):
            path = d.get("path")
            if not path or not app.allowed(path):
                raise PermissionError("Open this file with the Open button first.")
            data = Path(path).read_bytes()
            rot = d.get("rotation")
            rot = None if rot in (None, "", "auto") else int(rot)
            crop = str(d.get("crop", "1")).lower() in ("1", "true")
            plan = make_plan(data, 0, rot, crop)
            return path, data, plan

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._send(200, (HERE / "app.html").read_text(encoding="utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/ready":
                    startup_log(f"window ready (process running {_process_age()})")
                    return self._send(200, {"ok": True})
                if u.path == "/api/printers":
                    return self._send(200, printers())
                if u.path == "/api/update/check":
                    import updater
                    app.update_info = updater.check()
                    return self._send(200, app.update_info)
                if u.path == "/api/update/status":
                    return self._send(200, app.updater.status())
                if u.path == "/api/job":
                    return self._send(200, job_status(q["id"], q["printer"]))
                if u.path == "/api/state":
                    path, data, plan = self._plan_from({**q, "crop": "0"})
                    rot, crop, media, pages = page_state(data)
                    auto = make_plan(data, 0, None, True)
                    return self._send(200, {"rotation": rot, "status": status_of(data), "pages": pages,
                                            "autoRotation": auto["rotation"], "oversize": auto["oversize"]})
                if u.path == "/api/preview":
                    path, data, plan = self._plan_from(q)
                    new = apply_plan(data, plan) if (plan["crop"] or plan["rotation"] != page_state(data)[0]) else data
                    arr = _render_displayed(new, 0, 2.0)
                    from PIL import Image
                    buf = io.BytesIO()
                    Image.fromarray(arr).save(buf, "PNG", optimize=False, compress_level=1)
                    return self._send(200, buf.getvalue(), "image/png")
                return self._send(404, {"error": "Not found"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._send(400, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                d = self._json()
                if u.path == "/api/open":
                    if not app.choosing.acquire(blocking=False):
                        return self._send(200, {"files": [], "busy": True})
                    try:
                        picked = choose_files()
                    finally:
                        app.choosing.release()
                    files = []
                    for f in picked:
                        p = Path(f).resolve()
                        if p.suffix.lower() == ".pdf" and p.is_file():
                            app.opened.add(str(p))
                            files.append({"path": str(p), "name": p.stem})
                    return self._send(200, {"files": files})
                if u.path == "/api/save":
                    path, data, plan = self._plan_from(d)
                    ok, checks = save_in_place(path, plan)
                    return self._send(200, {"ok": ok, "checks": checks,
                                            "status": status_of(Path(path).read_bytes())})
                if u.path == "/api/print":
                    path, data, plan = self._plan_from(d)
                    if not d.get("printer"):
                        raise ValueError("Choose a printer first.")
                    changed = plan["crop"] or plan["rotation"] != page_state(data)[0]
                    new = apply_plan(data, plan) if changed else data
                    if changed:
                        ok, checks = verify(data, new, plan)
                        if not ok:
                            return self._send(200, {"ok": False, "checks": checks})
                    try:
                        p = printer_status(d["printer"])
                    except Exception:
                        p = lpstat_status(d["printer"])        # fallback when status can't be read
                    if p["problem"]:
                        return self._send(200, {"ok": False, "problem": p["problem"], "fix": p["fix"]})
                    copy = print_copy(new)
                    good, why = check_print_copy(new, copy)
                    if not good:
                        return self._send(200, {"ok": False, "problem": why})
                    pg = PdfReader(io.BytesIO(copy)).pages[0]
                    w_in, h_in = float(pg.mediabox.width) / PT, float(pg.mediabox.height) / PT
                    knows_media = bool(p.get("media") or p.get("media_default"))
                    media = pick_media(p, w_in, h_in) if knows_media else None
                    if knows_media and not media and plan["crop"] and plan["oversize"]:
                        media = pick_media(p, 4, 6)          # larger label: shrink onto 4 x 6 stock
                    if knows_media and not media:
                        have = media_size(p.get("media_default")) or ()
                        size = f"{have[0]:g} × {have[1]:g} in" if have else "a different size of"
                        hint = " Click Crop to 4×6 first." if not plan["crop"] and pick_media(p, 4, 6) else ""
                        return self._send(200, {"ok": False, "problem":
                            f"This page is {w_in:.2f} × {h_in:.2f} in, but the printer takes {size} "
                            f"labels.{hint}"})
                    job = submit(copy, d["printer"], Path(path).stem, plan["oversize"], media)
                    return self._send(200, {"ok": True, "job": job})
                if u.path == "/api/update/start":
                    if not app.update_info or not app.update_info.get("available"):
                        raise ValueError("Check for updates first.")
                    app.updater.start(app.update_info)
                    return self._send(200, app.updater.status())
                if u.path == "/api/fix-location":
                    import updater
                    return self._send(200, updater.fix_location(_quit_app))
                if u.path == "/api/update/install":
                    app.updater.install_and_quit(_quit_app)
                    return self._send(200, {"ok": True})
                if u.path == "/api/cancel":
                    cancel_job(d["job"])
                    return self._send(200, {"ok": True})
                if u.path == "/api/resume":
                    resume_printer(d["printer"])
                    p = printer_status(d["printer"])
                    return self._send(200, {"ok": not p["problem"], "problem": p["problem"]})
                return self._send(404, {"error": "Not found"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self._send(400, {"error": str(e)})

    return H


def _process_age():
    """How long since macOS started this process, including time before Python ran."""
    try:
        out = subprocess.run(["ps", "-o", "etime=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        return out + " since launch"
    except Exception:
        return "unknown"


class _LocalServer(ThreadingHTTPServer):
    """Python's server looks up this computer's network name when it starts (socket.getfqdn).
    On many Macs that lookup waits on DNS/Bonjour timeouts for several seconds on every launch.
    The app only talks to itself on 127.0.0.1, so skip it."""
    daemon_threads = True

    def server_bind(self):
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


def start_server(port=8737):
    app = App()
    for p in range(port, port + 50):
        try:
            srv = _LocalServer(("127.0.0.1", p), make_handler(app))
            break
        except OSError:
            continue
    else:
        sys.exit("No free port found.")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/"


def run_browser(port):
    url = start_server(port)
    print(f"Label Cropper is running at {url}\nClose this window to quit.")
    if os.environ.get("LABEL_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    while True:
        time.sleep(3600)


def run_window(port):
    global WINDOW
    startup_log("starting")
    threading.Thread(target=warm_up, daemon=True).start()
    import webview
    startup_log("window toolkit loaded")
    url = start_server(port)
    startup_log("server ready")
    WINDOW = webview.create_window("Label Cropper", url, width=1180, height=880, min_size=(720, 560),
                                   background_color="#2a2d31")
    webview.start(private_mode=False)


# ----------------------------------------------------------------------------
# self-test (used by the build to prove the packaged app works)
# ----------------------------------------------------------------------------
def _synthetic_label():
    """A letter page with a sideways 4x6-ish label in the top half, like eBay's, with no real data."""
    ops = ["0 0 0 RG 0 0 0 rg 1.4 w"]
    x0, y0, w, h = 100, 452, 400, 272                  # sideways label box
    ops.append(f"{x0} {y0} {w} {h} re S")
    for i in range(70):                                # barcode bars along the bottom of the turned label
        bw = 0.7 + (i * 13 % 4) * 0.5
        ops.append(f"{x0 + 20} {y0 + 30 + i * 3.4:.1f} 46 {bw:.2f} re f")
    for i in range(12):
        ops.append(f"{x0 + 200} {y0 + 40 + i * 18} 120 6 re f")
    content = ("\n".join(ops)).encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream"]
    out = io.BytesIO(); out.write(b"%PDF-1.4\n"); offs = []
    for i, o in enumerate(objs, 1):
        offs.append(out.tell()); out.write(b"%d 0 obj\n" % i + o + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for o in offs:
        out.write(b"%010d 00000 n \n" % o)
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref))
    return out.getvalue()


def selftest():
    d = tempfile.mkdtemp()
    p = Path(d) / "selftest-label.pdf"
    p.write_bytes(_synthetic_label())
    before = p.read_bytes()
    assert status_of(before) == "needs crop", status_of(before)
    plan = make_plan(before)
    ok, checks = save_in_place(p, plan)
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["name"] + ": " + c["detail"])
    after = p.read_bytes()
    assert ok and after.startswith(before) and status_of(after) == "cropped", "self-test failed"
    good, why = check_print_copy(after, print_copy(after))
    assert good, why
    pc = PdfReader(io.BytesIO(print_copy(after))).pages[0]
    assert [float(v) for v in pc.mediabox] == [0, 0, 288, 432] and int(pc.get("/Rotate", 0)) == 0
    print("PASS Printer copy: exact 4 x 6 page, upright, matches the preview")
    assert pick_media({"media": ["na_letter_8.5x11in", "na_index-4x6_4x6in"]}, 4, 6) == "na_index-4x6_4x6in"
    assert pick_media({"media": ["na_letter_8.5x11in"]}, 4, 6) is None
    # the window's server works on several threads at once; PDFium must survive that
    errors = []

    def hammer():
        try:
            for _ in range(15):
                pl = make_plan(before)
                nb = apply_plan(before, pl)
                verify(before, nb, pl)
                _render_displayed(nb, 0, 2.0)
        except Exception as e:
            errors.append(repr(e))
    workers = [threading.Thread(target=hammer) for _ in range(8)]
    [w.start() for w in workers]
    [w.join() for w in workers]
    assert not errors, f"parallel use failed: {errors[:2]}"
    print("PASS Parallel use: 8 threads x 15 crop/verify/render rounds with no errors")
    assert (HERE / "app.html").is_file(), "app.html missing from bundle"
    print("SELFTEST OK")


def _setup_logging():
    """Keep a log and a crash trace in ~/Library/Logs/Label Cropper (the app window has no console)."""
    import faulthandler
    import traceback
    logdir = Path.home() / "Library" / "Logs" / "Label Cropper"
    try:
        logdir.mkdir(parents=True, exist_ok=True)
        log = open(logdir / "label-cropper.log", "a", buffering=1, encoding="utf-8")
    except Exception:
        return
    cli = any(a in ("--selftest", "--update-selftest", "--startup-timing") for a in sys.argv)  # print to terminal
    if sys.stderr is None or (getattr(sys, "frozen", False) and not cli):
        sys.stderr = log
    if sys.stdout is None or (getattr(sys, "frozen", False) and not cli):
        sys.stdout = log
    faulthandler.enable(file=log, all_threads=True)
    log.write(f"\n--- started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    def hook(args):
        log.write("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
    threading.excepthook = hook


if __name__ == "__main__":
    _setup_logging()
    port = int(os.environ.get("LABEL_PORT", 8737))
    if "--startup-timing" in sys.argv:
        startup_log("python started")
        warm_up()
        url = start_server(port)
        startup_log("server ready")
        d = _synthetic_label()
        t = time.monotonic(); make_plan(d); status_of(d)
        startup_log(f"first label analysed in {time.monotonic() - t:.2f}s")
        startup_log(f"total (process running {_process_age()})")
    elif "--selftest" in sys.argv:
        selftest()
    elif "--update-selftest" in sys.argv:
        import updater
        i = sys.argv.index("--update-selftest")
        updater.selftest(sys.argv[i + 1], sys.argv[i + 2])
    elif "--browser" in sys.argv or os.environ.get("LABEL_BROWSER") == "1":
        run_browser(port)
    else:
        try:
            import webview  # noqa: F401
        except ImportError:
            run_browser(port)
        else:
            run_window(port)
