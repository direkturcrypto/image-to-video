"""E2E test v4: OpenAI gpt-image-2 request shape, retry, ordering, refs."""
import time, base64, io, zipfile
from pathlib import Path
import app as appmod

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

calls = []
fail_once = {"done": False}
class Resp:
    def __init__(self, code, payload=None, text="ok"):
        self.status_code=code; self._p=payload or {}; self.text=text; self.headers={}
    def json(self): return self._p

def fake_post(url, **kw):
    # generations = JSON body; edits = multipart (data+files)
    is_edit = url.endswith("/images/edits")
    n_imgs = len(kw.get("files", [])) if is_edit else 0
    calls.append({"url":url, "edit":is_edit, "imgs":n_imgs})
    # one transient 503 on the 3rd image call to exercise retry
    if "/images/" in url and len([c for c in calls if "/images/" in c["url"]])==3 and not fail_once["done"]:
        fail_once["done"]=True
        return Resp(503, text="overloaded")
    return Resp(200, {"data":[{"b64_json":base64.b64encode(PNG).decode()}]})

def fake_get(url, **kw):
    return Resp(200, {"data":[]})   # /v1/models

appmod.requests.post=fake_post
appmod.requests.get=fake_get

# IMPORTANT: redirect all I/O to a throwaway sandbox so running the test can
# never delete a user's real generated images in ./output (it used to wipe them).
import tempfile
_sandbox = Path(tempfile.mkdtemp(prefix="sketch_test_"))
appmod.OUTPUT_DIR = _sandbox / "output"
appmod.ANCHOR_DIR = _sandbox / "anchors"
appmod.FRAMES_DIR = _sandbox / "frames"
for d in (appmod.OUTPUT_DIR, appmod.ANCHOR_DIR, appmod.FRAMES_DIR):
    d.mkdir(parents=True, exist_ok=True)
appmod.PROJECT_FILE = appmod.OUTPUT_DIR / "project.json"
print("SANDBOX:", _sandbox)

client = appmod.app.test_client()

print("KEY TEST:", client.post("/api/test_key", json={"api_key":"sk-TEST"}).get_json())

# make 2 fake anchors -> these force the /edits endpoint
for i in range(2):
    (appmod.ANCHOR_DIR/f"anchor_{i}.jpg").write_bytes(PNG)

calls.clear()
r=client.post("/api/start", json={"api_key":"sk-TEST","prompts":[f"scene {i}" for i in range(4)],
    "settings":{"model":"gpt-image-2","retries":3,"use_previous":True,
                "delay":0,"quality":"low","size":"1024x1024",
                "style_suffix":"doodle","negative":"no 3d"}})
print("START:", r.get_json())
for _ in range(200):
    s=client.get("/api/status").get_json()
    if not s["running"]: break
    time.sleep(0.05)
print("FINAL:", {k:s[k] for k in ["running","total","done","error"]})

files=sorted(p.name for p in appmod.OUTPUT_DIR.glob("[0-9][0-9][0-9].png"))
print("FILES:", files)
assert files==[f"{i+1:03d}.png" for i in range(4)], "ordering wrong"

img_calls=[c for c in calls if "/images/" in c["url"]]
print("ENDPOINTS:", [("edit" if c["edit"] else "gen") for c in img_calls])
# with anchors present, all real calls should hit /edits
assert all(c["edit"] for c in img_calls), "should use /images/edits when refs present"
print("REF imgs per edit call:", [c["imgs"] for c in img_calls])
# first image: 2 anchors; later images: 2 anchors + prev = 3
assert img_calls[0]["imgs"]==2, "first should send 2 anchors"
assert any(c["imgs"]==3 for c in img_calls), "later should send anchors+prev"
print("RETRY happened:", fail_once["done"])
assert fail_once["done"], "retry path not exercised"

z=client.get("/api/zip")
names=sorted(zipfile.ZipFile(io.BytesIO(z.data)).namelist())
print("ZIP:", names)
assert "001.png" in names and "prompts.txt" in names

print("\nALL V4 TESTS PASSED ✓")
