"""
In-app updates for Label Cropper.

Where updates come from: the latest GitHub Release of daneazeeb-collab/label-cropper. Only the
repository owner and its own builds can publish releases there.

What happens on Update
  1. Download the installer for this Mac (Apple Silicon or Intel) with macOS's own curl.
  2. Check it against the SHA-256 fingerprint published with the release.
  3. Open the disk image and check the app inside: valid code signature, the same app
     identifier as this one, and the version the release says it is.
  4. Copy it next to the installed app, then quit. A small helper waits for the app to close,
     moves the current version to a backup, moves the new one into place and reopens it.
     If anything fails, the backup goes straight back.
Nothing is replaced until every check has passed.
"""
import hashlib
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

REPO = "daneazeeb-collab/label-cropper"
BUNDLE_ID = "com.danezeeb.labelcropper"
API_LATEST = os.environ.get("LABEL_UPDATE_FEED") or f"https://api.github.com/repos/{REPO}/releases/latest"
SUPPORT = Path.home() / "Library" / "Application Support" / "Label Cropper"
CACHE = Path.home() / "Library" / "Caches" / "Label Cropper" / "updates"
LOG = Path.home() / "Library" / "Logs" / "Label Cropper" / "label-cropper.log"

SWAP_SCRIPT = r"""#!/bin/bash
# Label Cropper update helper: swaps in the new app once the old one has quit.
PID="$1"; CUR="$2"; NEW="$3"; BAK="$4"; LOG="$5"; REOPEN="$6"
exec >>"$LOG" 2>&1
echo "update helper: waiting for Label Cropper (pid $PID) to quit"
alive() { kill -0 "$PID" 2>/dev/null && [ "$(ps -o stat= -p "$PID" 2>/dev/null | cut -c1)" != "Z" ]; }
for i in $(seq 1 150); do alive || break; sleep 0.2; done
if alive; then
  echo "update helper: app did not quit; update skipped"; rm -rf "$NEW"; exit 1
fi
mkdir -p "$(dirname "$BAK")"; rm -rf "$BAK"
if ! mv "$CUR" "$BAK"; then
  echo "update helper: could not move the current app aside; update skipped"
  rm -rf "$NEW"; [ "$REOPEN" = 1 ] && open "$CUR"; exit 1
fi
if ! mv "$NEW" "$CUR"; then
  echo "update helper: could not put the new app in place; restoring the previous version"
  mv "$BAK" "$CUR"; [ "$REOPEN" = 1 ] && open "$CUR"; exit 1
fi
echo "update helper: updated $CUR (previous version kept at $BAK)"
[ "$REOPEN" = 1 ] && open "$CUR"
exit 0
"""


def _log(msg):
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"updater: {msg}\n")
    except Exception:
        pass


def parse_version(v):
    nums = re.findall(r"\d+", v or "")
    return tuple(int(n) for n in nums[:4]) if nums else (0,)


def installed_app():
    """Path to the running Label Cropper.app, or None when not running as an installed app."""
    if os.environ.get("LABEL_TEST_APP"):                 # used for automated testing only
        return Path(os.environ["LABEL_TEST_APP"])
    if not getattr(sys, "frozen", False):
        return None
    p = Path(sys.executable).resolve()
    for parent in p.parents:
        if parent.suffix == ".app":
            return parent
    return None


def app_version(app=None):
    app = app or installed_app()
    if app is None:
        return os.environ.get("LABEL_VERSION", "dev")
    try:
        with open(app / "Contents" / "Info.plist", "rb") as f:
            return plistlib.load(f).get("CFBundleShortVersionString", "0")
    except Exception:
        return "0"


def this_arch():
    m = platform.machine().lower()
    if m == "x86_64":
        # an Intel build running under Rosetta on Apple Silicon should move to the native one
        try:
            r = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True)
            if r.stdout.strip() == "1":
                return "AppleSilicon"
        except Exception:
            pass
        return "Intel"
    return "AppleSilicon"


def _curl(args, timeout=60):
    r = subprocess.run(["curl", "-fsSL", "--retry", "2", "--connect-timeout", "15", *args],
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        msg = r.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Couldn't reach GitHub ({msg or 'curl exit ' + str(r.returncode)}). "
                           "Check your internet connection.")
    return r.stdout


def check(fetch=None):
    """What's the newest release, and is it newer than this app?"""
    cur = app_version()
    try:
        data = json.loads((fetch or (lambda: _curl([API_LATEST], 30)))())
    except Exception as e:
        return {"current": cur, "available": False, "error": str(e)}
    latest = (data.get("tag_name") or "").lstrip("v")
    assets = {a["name"]: a for a in data.get("assets", [])}
    want = this_arch()
    order = [want, "Intel" if want == "AppleSilicon" else "AppleSilicon"]
    pick = None
    for arch in order:
        dmg = assets.get(f"Label-Cropper-{arch}.dmg")
        sha = assets.get(f"Label-Cropper-{arch}.dmg.sha256")
        if dmg and sha:
            pick = {"arch": arch, "url": dmg["browser_download_url"], "size": dmg.get("size", 0),
                    "sha_url": sha["browser_download_url"]}
            break
    newer = parse_version(latest) > parse_version(cur) if cur != "dev" else False
    return {"current": cur, "latest": latest, "available": bool(newer and pick), "asset": pick,
            "notes": (data.get("body") or "").strip()[:2000], "published": data.get("published_at"),
            "installable": installed_app() is not None,
            "native": bool(pick and pick["arch"] == want)}


class Updater:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"phase": "idle"}
        self.ready = None             # (current_app, staged_app, backup_path)

    def status(self):
        with self.lock:
            return dict(self.state)

    def _set(self, **kw):
        with self.lock:
            self.state = {**self.state, **kw}

    def start(self, info):
        with self.lock:
            if self.state.get("phase") in ("downloading", "verifying", "installing"):
                return False
            self.state = {"phase": "downloading", "progress": 0, "version": info.get("latest")}
        threading.Thread(target=self._run, args=(info,), daemon=True).start()
        return True

    def _run(self, info):
        try:
            app = installed_app()
            if app is None:
                raise RuntimeError("Updates only work in the installed app.")
            staged = prepare(info, app, progress=lambda p: self._set(progress=p),
                             phase=lambda ph: self._set(phase=ph))
            self.ready = (app, staged, SUPPORT / "Previous Version" / app.name)
            self._set(phase="ready", progress=100)
        except Exception as e:
            _log(f"update failed: {e}")
            self._set(phase="error", error=str(e))

    def install_and_quit(self, quit_fn, reopen=True):
        if not self.ready:
            raise RuntimeError("No update is ready to install.")
        cur, staged, backup = self.ready
        launch_swap(cur, staged, backup, reopen=reopen)
        self._set(phase="installing")
        _log(f"installing update, quitting so it can be swapped in")
        threading.Timer(0.4, quit_fn).start()


def prepare(info, app, progress=lambda p: None, phase=lambda p: None, download=None):
    """Download, verify and stage the new app next to the current one. Returns the staged path."""
    asset = info.get("asset") or {}
    if not asset:
        raise RuntimeError("This release has no installer for this Mac.")
    if not os.access(app.parent, os.W_OK):
        raise RuntimeError(f"The app can't write to {app.parent}. Install this update by hand: "
                           "open the .dmg and drag Label Cropper into Applications.")
    CACHE.mkdir(parents=True, exist_ok=True)
    dmg = CACHE / f"Label-Cropper-{info['latest']}-{asset['arch']}.dmg"
    want_sha = _expected_sha(asset)
    if not (dmg.exists() and _sha256(dmg) == want_sha):
        part = dmg.with_suffix(".part")
        part.unlink(missing_ok=True)
        if download:
            download(asset, part)
        else:
            _download(asset, part, progress)
        part.rename(dmg)

    phase("verifying")
    got = _sha256(dmg)
    if got != want_sha:
        dmg.unlink(missing_ok=True)
        raise RuntimeError("The download didn't match its published fingerprint, so it was thrown away. "
                           "Nothing was changed.")
    staged = app.parent / f".{app.stem}.update.app"
    shutil.rmtree(staged, ignore_errors=True)
    mnt = Path(tempfile.mkdtemp(prefix="labelcropper-update-"))
    try:
        _run(["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mnt), str(dmg)],
             "Couldn't open the downloaded installer.")
        try:
            src = mnt / app.name
            if not src.is_dir():
                cand = [p for p in mnt.iterdir() if p.suffix == ".app"]
                if not cand:
                    raise RuntimeError("The installer doesn't contain Label Cropper.")
                src = cand[0]
            check_app(src, info["latest"])
            _run(["ditto", str(src), str(staged)], "Couldn't copy the new version.")
        finally:
            subprocess.run(["hdiutil", "detach", str(mnt), "-quiet", "-force"], capture_output=True)
    finally:
        shutil.rmtree(mnt, ignore_errors=True)
    check_app(staged, info["latest"])
    subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(staged)], capture_output=True)
    _log(f"staged {info['latest']} ({asset['arch']}) at {staged}")
    return staged


def check_app(app, version):
    try:
        with open(app / "Contents" / "Info.plist", "rb") as f:
            pl = plistlib.load(f)
    except Exception:
        raise RuntimeError("The new version is damaged (its app information can't be read). Nothing was changed.")
    if pl.get("CFBundleIdentifier") != BUNDLE_ID:
        raise RuntimeError("The download isn't Label Cropper. Nothing was changed.")
    if parse_version(pl.get("CFBundleShortVersionString")) != parse_version(version):
        raise RuntimeError(f"The download says it's version {pl.get('CFBundleShortVersionString')}, "
                           f"not {version}. Nothing was changed.")
    r = subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("The new version's code signature is broken, so it wasn't installed. Nothing was changed.")


def launch_swap(cur, staged, backup, reopen=True):
    SUPPORT.mkdir(parents=True, exist_ok=True)
    script = SUPPORT / "apply-update.sh"
    script.write_text(SWAP_SCRIPT)
    script.chmod(0o755)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(["/bin/bash", str(script), str(os.getpid()), str(cur), str(staged), str(backup),
                      str(LOG), "1" if reopen else "0"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, close_fds=True)


def _expected_sha(asset):
    text = _curl([asset["sha_url"]], 30).decode(errors="replace")
    m = re.search(r"\b([0-9a-f]{64})\b", text.lower())
    if not m:
        raise RuntimeError("The release's fingerprint file is unreadable, so the update was skipped.")
    return m.group(1)


def _download(asset, dest, progress):
    total = asset.get("size") or 0
    p = subprocess.Popen(["curl", "-fsSL", "--retry", "2", "--connect-timeout", "15", "-o", str(dest), asset["url"]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    while p.poll() is None:
        try:
            if total:
                progress(min(99, int(dest.stat().st_size * 100 / total)))
        except FileNotFoundError:
            pass
        try:
            p.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            pass
    if p.returncode != 0:
        err = p.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"The download stopped ({err or 'curl exit ' + str(p.returncode)}). Nothing was changed.")
    progress(100)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, err):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{err} ({(r.stderr or r.stdout).strip()[:200]})")
    return r.stdout


def selftest(dmg, version):
    """Build-machine test: install an 'old' copy of the app, then update it from `dmg` for real."""
    import time
    root = Path(tempfile.mkdtemp(prefix="lc-update-test-"))
    apps = root / "Applications"
    apps.mkdir()
    mnt = root / "mnt"
    mnt.mkdir()
    _run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mnt), str(dmg)], "attach")
    try:
        _run(["ditto", str(mnt / "Label Cropper.app"), str(apps / "Label Cropper.app")], "copy")
    finally:
        subprocess.run(["hdiutil", "detach", str(mnt), "-force"], capture_output=True)
    old = apps / "Label Cropper.app"
    with open(old / "Contents" / "Info.plist", "rb") as f:
        pl = plistlib.load(f)
    pl["CFBundleShortVersionString"] = "1.0.0"
    with open(old / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(pl, f)
    (old / "Contents" / "OLD-MARKER").write_text("old")

    sha = _sha256(Path(dmg))
    shafile = root / "x.sha256"
    shafile.write_text(f"{sha}  Label-Cropper.dmg\n")
    info = {"latest": version, "asset": {"arch": "test", "url": "file://" + str(Path(dmg).resolve()),
                                          "sha_url": "file://" + str(shafile), "size": Path(dmg).stat().st_size}}
    global CACHE, SUPPORT
    CACHE, SUPPORT = root / "cache", root / "support"
    staged = prepare(info, old)
    print("PASS Update staged and verified:", staged.name)

    # a tampered download must be refused
    bad = dict(info, asset=dict(info["asset"], sha_url="file://" + str(root / "bad.sha256")))
    (root / "bad.sha256").write_text("0" * 64 + "\n")
    shutil.rmtree(CACHE, ignore_errors=True)
    try:
        prepare(bad, old)
        raise AssertionError("tampered download was accepted")
    except RuntimeError as e:
        print("PASS Tampered download refused:", e)

    # run the real swap helper against a stand-in process that then exits
    staged = prepare(info, old)
    stand_in = subprocess.Popen(["sleep", "1"])
    script = SUPPORT / "apply-update.sh"
    SUPPORT.mkdir(parents=True, exist_ok=True)
    script.write_text(SWAP_SCRIPT)
    log = root / "update.log"
    subprocess.run(["/bin/bash", str(script), str(stand_in.pid), str(old), str(staged),
                    str(SUPPORT / "Previous Version" / old.name), str(log), "0"], check=True)
    print(log.read_text().strip())
    assert app_version(old) == version, f"installed version is {app_version(old)}"
    assert not (old / "Contents" / "OLD-MARKER").exists()
    assert (SUPPORT / "Previous Version" / old.name / "Contents" / "OLD-MARKER").exists()
    assert not staged.exists()
    _run(["codesign", "--verify", "--deep", "--strict", str(old)], "installed app signature")
    r = subprocess.run([str(old / "Contents" / "MacOS" / "Label Cropper"), "--selftest"], capture_output=True, text=True)
    assert r.returncode == 0 and "SELFTEST OK" in r.stdout, f"exit {r.returncode}: {r.stdout[-800:]} {r.stderr[-800:]}"
    print(f"PASS Updated 1.0.0 -> {version}; previous version kept; updated app passes its self-test")
    shutil.rmtree(root, ignore_errors=True)
    print("UPDATE SELFTEST OK")
