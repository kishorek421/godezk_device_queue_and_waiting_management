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
import threading
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

# Tracking / wait-time tuning
MIN_QUEUE_HITS         = int(os.environ.get("MIN_QUEUE_HITS", "3"))
MIN_TRACK_IOU          = float(os.environ.get("MIN_TRACK_IOU", "0.2"))
MAX_TRACK_DISTANCE     = float(os.environ.get("MAX_TRACK_DISTANCE", "150.0"))
MAX_TRACK_AGE          = int(os.environ.get("MAX_TRACK_AGE", "5"))
MAX_TRACK_AGE_SECONDS  = float(os.environ.get("MAX_TRACK_AGE_SECONDS", "5.0"))
INFERENCE_IMGSZ        = int(os.environ.get("INFERENCE_IMGSZ", "640"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Metrics ───────────────────────────────────────────────────
detection_requests  = Counter("queue_detection_requests_total",  "Total detection requests")
queue_violations    = Counter("queue_limit_exceeded_total",      "Total queue limit violations")
detection_latency   = Histogram("queue_detection_latency_seconds", "Detection latency")

# ── Person Tracker ────────────────────────────────────────────

def _centroid(bbox: tuple) -> tuple:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _iou(box_a: tuple, box_b: tuple) -> float:
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid_distance(c_a: tuple, c_b: tuple) -> float:
    return float(np.hypot(c_a[0] - c_b[0], c_a[1] - c_b[1]))


class Track:
    """A single tracked person with queue/wait state."""
    __slots__ = (
        "track_id", "bbox", "centroid", "confidence",
        "hits", "misses", "state",
        "first_seen", "last_seen",
        "is_inside", "inside_hits", "first_inside_time",
        "entry_time", "wait_time"
    )

    def __init__(self, track_id: int, bbox: tuple, confidence: float, now: float):
        self.track_id = track_id
        self.bbox = bbox
        self.centroid = _centroid(bbox)
        self.confidence = confidence
        self.hits = 1
        self.misses = 0
        self.state = "PENDING"
        self.first_seen = now
        self.last_seen = now
        self.is_inside = False
        self.inside_hits = 0
        self.first_inside_time: Optional[float] = None
        self.entry_time: Optional[float] = None
        self.wait_time = 0.0

    def bottom_mid(self) -> tuple:
        return (int(self.centroid[0]), int(self.bbox[3]))

    def update(self, bbox: tuple, confidence: float, now: float):
        self.bbox = bbox
        self.centroid = _centroid(bbox)
        self.confidence = confidence
        self.last_seen = now
        self.misses = 0
        self.hits += 1

    def mark_missed(self):
        self.misses += 1

    def is_dead(self, now: float, max_age: int, max_age_seconds: float) -> bool:
        return self.misses > max_age or (now - self.last_seen) > max_age_seconds

    def update_queue_state(self, now: float, inside: bool, min_hits: int) -> float:
        completed_wait = 0.0
        if inside:
            self.inside_hits += 1
            if self.first_inside_time is None:
                self.first_inside_time = now
            if self.state != "IN_QUEUE" and self.inside_hits >= min_hits:
                self.state = "IN_QUEUE"
                self.entry_time = self.first_inside_time
            self.is_inside = True
        else:
            self.inside_hits = 0
            self.first_inside_time = None
            if self.state == "IN_QUEUE":
                completed_wait = now - self.entry_time
                self.state = "CONFIRMED_OUTSIDE"
            self.is_inside = False
        self.refresh_wait_time(now)
        return completed_wait

    def refresh_wait_time(self, now: float):
        if self.state == "IN_QUEUE" and self.entry_time is not None:
            self.wait_time = now - self.entry_time
        else:
            self.wait_time = 0.0

    def complete_wait(self, now: float) -> float:
        if self.state == "IN_QUEUE" and self.entry_time is not None:
            return now - self.entry_time
        return 0.0


class PersonTracker:
    """Person tracker using IoU + centroid distance with stateful queue/wait logic."""
    def __init__(
        self,
        min_hits: int = MIN_QUEUE_HITS,
        min_iou: float = MIN_TRACK_IOU,
        max_distance: float = MAX_TRACK_DISTANCE,
        max_age: int = MAX_TRACK_AGE,
        max_age_seconds: float = MAX_TRACK_AGE_SECONDS
    ):
        self.next_track_id = 0
        self.tracks: List[Track] = []
        self.min_hits = min_hits
        self.min_iou = min_iou
        self.max_distance = max_distance
        self.max_age = max_age
        self.max_age_seconds = max_age_seconds

    def active_tracks(self) -> List[Track]:
        return self.tracks

    def update(self, detections: List[tuple], now: float) -> List[Track]:
        # detections: list of ((x1,y1,x2,y2), confidence)
        matched_track = set()
        matched_det = set()

        candidates = []
        for ti, track in enumerate(self.tracks):
            for di, (bbox, conf) in enumerate(detections):
                iou = _iou(track.bbox, bbox)
                dist = _centroid_distance(track.centroid, _centroid(bbox))
                if iou >= self.min_iou:
                    quality = iou + 1.0
                elif dist <= self.max_distance:
                    quality = 1.0 - (dist / (self.max_distance or 1.0))
                else:
                    continue
                candidates.append((quality, ti, di))

        candidates.sort(reverse=True)

        for _, ti, di in candidates:
            if ti in matched_track or di in matched_det:
                continue
            track = self.tracks[ti]
            det = detections[di]
            track.update(det[0], det[1], now)
            matched_track.add(ti)
            matched_det.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track:
                track.mark_missed()

        removed = [track for track in self.tracks if track.is_dead(now, self.max_age, self.max_age_seconds)]
        self.tracks = [track for track in self.tracks if not track.is_dead(now, self.max_age, self.max_age_seconds)]

        for di, (bbox, conf) in enumerate(detections):
            if di not in matched_det:
                new_track = Track(self.next_track_id, bbox, conf, now)
                self.next_track_id += 1
                self.tracks.append(new_track)

        return removed


# ── Global State ──────────────────────────────────────────────
_model = None
_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Per-camera trackers and wait times
_trackers: Dict[str, PersonTracker] = {}
_completed_waits: Dict[str, List[float]] = {}  # camera_id -> list of completed wait durations
_camera_locks: Dict[str, threading.Lock] = {}
_state_lock = threading.Lock()
_inference_lock = threading.Lock()

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
    with _inference_lock:
        results = _model.predict(
            img,
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
            verbose=False,
            imgsz=INFERENCE_IMGSZ
        )

    now = time.monotonic()
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().int().tolist()
        confs = boxes.conf.cpu().tolist()
        for bbox, conf in zip(xyxy, confs):
            detections.append(((bbox[0], bbox[1], bbox[2], bbox[3]), float(conf)))

    with _state_lock:
        if camera_id not in _trackers:
            _trackers[camera_id] = PersonTracker()
            _completed_waits[camera_id] = []
            _camera_locks[camera_id] = threading.Lock()
        camera_lock = _camera_locks[camera_id]

    with camera_lock:
        tracker = _trackers[camera_id]
        completed_list = _completed_waits[camera_id]

        removed = tracker.update(detections, now)
        for track in removed:
            completed_wait = track.complete_wait(now)
            if completed_wait > 0:
                completed_list.append(completed_wait)
                if len(completed_list) > 100:
                    completed_list.pop(0)
                logger.info(f"[{camera_id}] Track {track.track_id} removed from queue. Wait: {completed_wait:.2f}s.")

        roi_polygon = get_roi_for_camera(camera_id)
        persons = []
        in_queue = []

        for track in tracker.active_tracks():
            if track.misses == 0:
                inside = _is_inside_roi(track.bottom_mid(), roi_polygon)
                completed = track.update_queue_state(now, inside, MIN_QUEUE_HITS)
                if completed > 0:
                    completed_list.append(completed)
                    if len(completed_list) > 100:
                        completed_list.pop(0)
                    logger.info(f"[{camera_id}] Track {track.track_id} left queue zone. Wait: {completed:.2f}s.")
            else:
                # Keep wait time running for briefly missed queue members.
                track.refresh_wait_time(now)

            # Only draw visible tracks; queue count still includes briefly missed members.
            if track.misses == 0:
                persons.append(PersonBBox(
                    x1=int(track.bbox[0]),
                    y1=int(track.bbox[1]),
                    x2=int(track.bbox[2]),
                    y2=int(track.bbox[3]),
                    confidence=track.confidence,
                    track_id=track.track_id,
                    wait_time_seconds=track.wait_time,
                    inside_roi=track.is_inside
                ))

            if track.state == "IN_QUEUE":
                in_queue.append(track.wait_time)

        queue_length = len(in_queue)
        max_wait = max(in_queue) if in_queue else 0.0
        avg_wait = np.mean(in_queue) if in_queue else 0.0

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

    dep_id = request.headers.get("x-deployment-id")
    if dep_id:
        camera_id = f"{dep_id}:{camera_id}"

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

@app.post("/reset")
async def reset(request: Request):
    """Reset tracker state for a specific camera, deployment, or all trackers."""
    cam_id = request.headers.get("x-camera-id")
    dep_id = request.headers.get("x-deployment-id")

    try:
        body = await request.json()
        cam_id = body.get("camera_id") or cam_id
        dep_id = body.get("deployment_id") or dep_id
    except Exception:
        pass

    target_key = None
    if dep_id and cam_id:
        target_key = f"{dep_id}:{cam_id}"
    elif cam_id:
        target_key = cam_id
    elif dep_id:
        target_key = dep_id

    with _state_lock:
        if target_key:
            keys_to_del = [k for k in list(_trackers.keys()) if k == target_key or k.startswith(f"{target_key}:") or k.endswith(f":{target_key}")]
            for k in keys_to_del:
                _trackers.pop(k, None)
                _completed_waits.pop(k, None)
                _camera_locks.pop(k, None)
            logger.info(f"Reset trackers for target_key='{target_key}' ({len(keys_to_del)} keys removed)")
            return {"success": True, "removed_keys": keys_to_del}
        else:
            count = len(_trackers)
            _trackers.clear()
            _completed_waits.clear()
            _camera_locks.clear()
            logger.info(f"Reset ALL trackers ({count} cameras cleared)")
            return {"success": True, "cleared_all": count}

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
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1, loop="asyncio")
