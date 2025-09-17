# 🎬 VideoLens Pro – AI-Powered Video Analysis Pipeline

> Drop a video, get instant Hollywood-grade insights.  
> 100 % local, 100 % open-source, zero configuration.

---

## ✨ What it does
1. Drag-and-drop any video (≤ 500 MB)  
2. Real-time progress bar while the AI pipeline runs  
3. Receive a complete JSON report + 5 smart thumbnails  
4. Download everything or embed the API in your own product

---

## 🧠 Analysis super-powers
| Step | What you get |
|---|---|
| **Basic Info** | duration, resolution, fps, frame-count, file-size, bitrate |
| **Frame Analysis** | average brightness, motion score, stability rating |
| **Scene Detection** | automatic scene cuts, average scene length, complexity rating |
| **Thumbnails** | 5 key-frames (10 %, 25 %, 50 %, 75 %, 90 %) ready for CDN |
| **Quality Metrics** | visual quality, compression efficiency, stability index |

---

## 🚀 30-second start
```
bash
# 1. clone repo (or just copy the two files)
git clone https://github.com/yourname/videolens-pro.git
cd videolens-pro

# 2. install deps
pip install fastapi uvicorn opencv-python moviepy aiofiles

# 3. run backend
python main.py
# → API now on http://localhost:8000

# 4. open frontend
# just double-click `index.html` or serve it via any static server

Visit [http://localhost:8000](http://localhost:8000) – the interactive docs (Swagger) are live.
```
---

## 🔌 API cheat-sheet
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-video` | multipart upload, returns `job_id` |
| `GET`  | `/job-status/{job_id}` | progress & status |
| `GET`  | `/results/{job_id}` | full JSON results |
| `GET`  | `/download-results/{job_id}` | download JSON file |
| `GET`  | `/thumbnail/{filename}` | serve generated thumbnail |
| `GET`  | `/jobs` | list all active jobs |
| `DELETE`| `/job/{job_id}` | purge job + files |

---

## 📁 Project layout
```
videolens-pro/
├── main.py          # FastAPI backend (self-contained)
├── index.html       # drop-dead gorgeous frontend (no build step)
├── uploads/         # temp video storage (auto-created)
├── results/         # JSON + thumbnails (auto-created)
└── temp/            # scratch workspace (auto-created)
```
Everything is cleaned up when you call `DELETE /job/{job_id}`.

---

## 🎨 Frontend highlights
* 100 % vanilla ES6 – no frameworks, no npm install  
* Progressive-web-app ready (responsive, offline-capable shell)  
* Drag-&-drop + click-to-browse  
* Live progress bar with emoji flair  
* Auto-scrolls to results when ready  
* Keyboard shortcut: `Ctrl + U` to open file picker  

---

## 🛠️ Customising the pipeline
Open `main.py` → class `VideoPipeline`.  
Each `step_*` method is `async` and self-contained; add your own:

```python
async def step_6_face_detection(self):
    await self.update_status("face_detection", 85, "Detecting faces…")
    # your logic here
    self.results["faces"] = face_data
```

Restart the server – new step appears automatically in the progress bar.

---

## 🐳 Docker one-liner
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build & run:  
```bash
docker build -t videolens-pro .
docker run -p 8000:8000 -v $(pwd)/results:/app/results videolens-pro
```

---

## 🔒 Production checklist
* Replace in-memory `job_status` dict with Redis or PostgreSQL  
* Set `allow_origins` to your real domain instead of `["*"]`  
* Run behind HTTPS (Traefik, Caddy, Nginx)  
* Add auth middleware if you expose the API publicly  
* Increase payload limit in FastAPI if you need > 500 MB  
* Use a GPU-enabled OpenCV build for 10× speed-up

---

## 📄 Sample output snippet
```json
{
  "basic_info": {
    "duration_seconds": 127.6,
    "resolution": { "width": 1920, "height": 1080 },
    "fps": 30.0,
    "frame_count": 3828,
    "file_size_mb": 42.3,
    "bitrate_kbps": 2789
  },
  "scene_detection": {
    "total_scenes": 14,
    "average_scene_duration": 9.1,
    "scenes": [ … ]
  },
  "thumbnails": [
    {
      "filename": "a1b2c3d4_thumb_0.jpg",
      "timestamp": 12.7,
      "frame_number": 382
    }
  ]
}
```

---

## 📄 License
MIT – do whatever you want, just don’t blame us.

---

## 🙋‍♂️ Contributing
PRs welcome!  
Road-map: audio analysis, OCR on frames, object detection, GPU acceleration.

---

**Star ⭐ if this saved you a day of coding.**  
Made with ❤️ by the community, for the community.
```