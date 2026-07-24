"""
GoDezk Queue & Wait Time Analytics Service
YOLOv11-based queue tracker — standalone HTTP API, workflow node
Trained on COCO / custom person tracking, runs inference on CPU/GPU
"""
import os
import sys
import json
import asyncio
import base64
import logging
import time
import concurrent.futures
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

import numpy as np
import cv2
from ultralytics import YOLO

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

# ── Config ────────────────────────────────────────────────────
MODEL_PATH           = os.environ.get("MODEL_PATH", "../queue_management/models/yolo11s.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.45"))
LOG_LEVEL            = os.environ.get("LOG_LEVEL", "INFO")

# Default ROI: whole frame if not specified
DEFAULT_ROI = [[0, 0], [4096, 0], [4096, 4096], [0, 4096]]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────
detection_requests  = Counter("queue_detection_requests_total",  "Total detection requests")
queue_violations    = Counter("queue_limit_exceeded_total",      "Total queue limit violations")
detection_latency   = Histogram("queue_detection_latency_seconds", "Detection latency")

# ── Centroid Tracker ──────────────────────────────────────────
class CentroidTracker:
    """Centroid tracker to identify individuals across frames."""
    def __init__(self, max_disappeared: int = 15):
        self.next_object_id = 0
        self.objects: Dict[int, tuple] = {}  # object_id -> (cX, cY, x1, y1, x2, y2)
        self.disappeared: Dict[int, int] = {}  # object_id -> frame_count
        self.max_disappeared = max_disappeared

    def register(self, bbox: tuple, centroid: tuple) -> int:
        self.objects[self.next_object_id] = (centroid[0], centroid[1], bbox[0], bbox[1], bbox[2], bbox[3])
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id: int):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects: List[tuple]) -> Dict[int, tuple]:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(rects[i], input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [[val[0], val[1]] for val in self.objects.values()]

            # Compute Euclidean distances
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (
                    input_centroids[col][0], input_centroids[col][1],
                    rects[col][0], rects[col][1], rects[col][2], rects[col][3]
                )
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(rects[col], input_centroids[col])

        return self.objects


# ── Global State ──────────────────────────────────────────────
_model = None
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Per-camera trackers and wait times
_trackers: Dict[str, CentroidTracker] = {}
_active_queues: Dict[str, Dict[int, float]] = {}  # camera_id -> {track_id -> entry_timestamp}
_completed_waits: Dict[str, List[float]] = {}  # camera_id -> list of wait durations

# ── App ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application):
    global _model
    logger.info("Loading YOLO model...")
    loop = asyncio.get_event_loop()
    
    # Check fallback path if local models/ path does not exist
    resolved_model_path = MODEL_PATH
    if not os.path.exists(resolved_model_path):
        fallback_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "desk_management_system", "models", "yolo11s.pt")
        if os.path.exists(fallback_path):
            resolved_model_path = fallback_path
        else:
            # Check current models dir
            local_path = os.path.join("models", "yolo11s.pt")
            if os.path.exists(local_path):
                resolved_model_path = local_path

    logger.info(f"Model path resolved to: {resolved_model_path}")
    _model = await loop.run_in_executor(_thread_pool, _load_model, resolved_model_path)
    logger.info("GoDezk Queue & Wait Time Analytics Service ready")
    yield

app = FastAPI(title="GoDezk Queue & Wait Time Analytics Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────

class DetectRequest(BaseModel):
    image_b64: str
    camera_id: Optional[str] = None
    frame_id:  Optional[str] = None

class PersonBBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int
    wait_time_seconds: float
    inside_roi: bool

class DetectResponse(BaseModel):
    success:            bool
    queue_length:       int
    average_wait_time:  float  # in seconds
    max_wait_time:      float  # in seconds
    camera_id:          Optional[str] = None
    frame_id:           Optional[str] = None
    persons:            List[PersonBBox]
    processing_time_ms: float

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status:       str
    model_loaded: bool
    timestamp:    float


# ── Helpers ───────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def _load_model(path: str) -> YOLO:
    m = YOLO(path)
    # Warmup prediction
    dummy = np.zeros((300, 300, 3), dtype=np.uint8)
    m.predict(dummy, classes=[0], verbose=False)
    logger.info(f"YOLO model loaded successfully from {path}")
    return m

def get_roi_for_camera(camera_id: str) -> np.ndarray:
    raw_roi = os.environ.get("QUEUE_ROI")
    if not raw_roi:
        return np.array(DEFAULT_ROI, dtype=np.int32)
    try:
        data = json.loads(raw_roi)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):  # Simple list of points
                return np.array(data, dtype=np.int32)
        elif isinstance(data, dict):
            # Dict mapping camera_id to points
            if camera_id in data:
                return np.array(data[camera_id], dtype=np.int32)
            elif "default" in data:
                return np.array(data["default"], dtype=np.int32)
    except Exception as e:
        logger.error(f"Failed to parse QUEUE_ROI: {e}")
    return np.array(DEFAULT_ROI, dtype=np.int32)

def _is_inside_roi(point: tuple, roi_polygon: np.ndarray) -> bool:
    # point: (x, y)
    return cv2.pointPolygonTest(roi_polygon, (float(point[0]), float(point[1])), False) >= 0

def _run_inference(img: np.ndarray, camera_id: str):
    """Run YOLO prediction and update tracking/wait times."""
    results = _model.predict(img, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
    
    rects = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().int().tolist()
        confs = boxes.conf.cpu().tolist()
        for bbox, conf in zip(xyxy, confs):
            rects.append((bbox[0], bbox[1], bbox[2], bbox[3]))

    # Retrieve or create tracker for this camera
    if camera_id not in _trackers:
        _trackers[camera_id] = CentroidTracker(max_disappeared=15)
        _active_queues[camera_id] = {}
        _completed_waits[camera_id] = []

    tracker = _trackers[camera_id]
    active_queue = _active_queues[camera_id]
    completed_list = _completed_waits[camera_id]
    
    tracked_objects = tracker.update(rects)
    current_time = time.time()
    
    roi_polygon = get_roi_for_camera(camera_id)
    
    # Process tracked objects
    persons = []
    current_tracked_ids = set()
    
    for object_id, info in tracked_objects.items():
        cX, cY, x1, y1, x2, y2 = info
        current_tracked_ids.add(object_id)
        
        # Use bottom-middle point of the bounding box to test ROI presence (more standard for queues)
        bottom_mid_point = (int((x1 + x2) / 2), y2)
        inside = _is_inside_roi(bottom_mid_point, roi_polygon)
        
        # Handle entering queue
        if inside:
            if object_id not in active_queue:
                active_queue[object_id] = current_time
                logger.info(f"[{camera_id}] Track ID {object_id} entered the queue zone.")
            
            wait_time = current_time - active_queue[object_id]
        else:
            # If outside but was previously in queue, trigger a queue leave
            if object_id in active_queue:
                wait_duration = current_time - active_queue[object_id]
                completed_list.append(wait_duration)
                if len(completed_list) > 100:  # Keep list bounded
                    completed_list.pop(0)
                logger.info(f"[{camera_id}] Track ID {object_id} exited queue zone. Wait duration: {wait_duration:.2f}s.")
                del active_queue[object_id]
            
            wait_time = 0.0

        # Confidence is calculated by matching original rects if needed, or default as general
        # Find corresponding prediction confidence score
        confidence = 0.90
        for rect, conf in zip(rects, [b.conf.cpu().item() for r in results for b in r.boxes]):
            if rect == (x1, y1, x2, y2):
                confidence = conf
                break

        persons.append(PersonBBox(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=confidence,
            track_id=object_id,
            wait_time_seconds=wait_time,
            inside_roi=inside
        ))

    # Deregister/cleanup active queue items that are no longer tracked
    for object_id in list(active_queue.keys()):
        if object_id not in current_tracked_ids:
            # Person left camera entirely
            wait_duration = current_time - active_queue[object_id]
            completed_list.append(wait_duration)
            if len(completed_list) > 100:
                completed_list.pop(0)
            logger.info(f"[{camera_id}] Track ID {object_id} left tracking. Total wait duration: {wait_duration:.2f}s.")
            del active_queue[object_id]

    # Calculate metrics
    queue_length = len(active_queue)
    
    current_wait_times = [current_time - start_t for start_t in active_queue.values()]
    max_wait = max(current_wait_times) if current_wait_times else 0.0
    
    avg_wait = np.mean(completed_list) if completed_list else (np.mean(current_wait_times) if current_wait_times else 0.0)

    return int(queue_length), float(avg_wait), float(max_wait), persons


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if _model is not None else "starting",
        model_loaded=_model is not None,
        timestamp=time.time(),
    )

@app.get("/alive")
async def alive():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "alive"}

@app.get("/ready")
async def ready():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/detect", response_model=DetectResponse)
async def detect(request: Request):
    """Detect queue size and wait times.
    Accepts binary JPEG (Content-Type: image/jpeg) or JSON {image_b64, camera_id?, frame_id?}."""
    detection_requests.inc()
    start = time.time()

    content_type = request.headers.get("content-type", "application/json")
    if content_type.startswith("image/"):
        raw       = await request.body()
        img       = _decode_image(raw)
        camera_id = request.headers.get("x-camera-id", "default")
        frame_id  = request.headers.get("x-frame-id", str(time.time()))
    else:
        body      = await request.json()
        req       = DetectRequest(**body)
        img       = _decode_image(base64.b64decode(req.image_b64))
        camera_id = req.camera_id or "default"
        frame_id  = req.frame_id or str(time.time())

    try:
        loop = asyncio.get_event_loop()
        queue_len, avg_wait, max_wait, persons = await loop.run_in_executor(
            _thread_pool, _run_inference, img, camera_id
        )

        elapsed = time.time() - start
        detection_latency.observe(elapsed)

        return DetectResponse(
            success=True,
            queue_length=queue_len,
            average_wait_time=avg_wait,
            max_wait_time=max_wait,
            camera_id=camera_id,
            frame_id=frame_id,
            persons=persons,
            processing_time_ms=elapsed * 1000,
        )
    except Exception as e:
        logger.error(f"Queue detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    if sys.platform != "win32":
        try:
            import uvloop  # type: ignore
            uvloop.install()
        except ImportError:
            pass
    uvicorn.run(app, host="0.0.0.0", port=8014, workers=1, loop="asyncio")
