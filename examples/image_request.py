"""Ask a yes/no question about a photo.

    python examples/image_request.py photo.jpg [--url http://localhost:8090]

Images need a vision-capable engine: vLLM 0.31 or later, transformers, llama.cpp with the model's mmproj file, or MLX.
"""
import argparse
import base64
import mimetypes

from client import decide

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--url", default="http://localhost:8090")
a = ap.parse_args()
mime = mimetypes.guess_type(a.image)[0] or "image/jpeg"
with open(a.image, "rb") as f:
    uri = f"data:{mime};base64," + base64.b64encode(f.read()).decode()
out = decide(a.url, {
    "state": "Photo taken at delivery: <image>",          # <image> marks where the image goes
    "questions": {"damaged": {"type": "noul", "instructions": "Is the parcel damaged?"}},
    "images": [uri],
})
print(f"P(damaged) = {out['answers']['damaged']['noul']:.2f}")
