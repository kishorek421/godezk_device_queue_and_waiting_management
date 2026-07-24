# GoDezk Queue & Wait Time Analytics Service

Standalone API for real-time queue length monitoring and individual wait-time analysis using YOLOv11 person detection and centroid-based object tracking.

## Features
- **FastAPI HTTP Endpoints:** `/health`, `/alive`, `/ready`, `/detect`, `/metrics`.
- **Zoning (ROI):** Check if detected people are inside a specific queue polygon zone (defined in coordinates).
- **Wait Time Tracking:** Centroid tracking keeps persistent track IDs across frames per camera to count exact queue duration.
- **Prometheus Metrics:** Integrated metrics for latency, requests, and queue limit violations.

## API Usage

### 1. Health Checks
- `GET /health`
- `GET /alive`
- `GET /ready`

### 2. Detection
`POST /detect`
- **Headers:** `Content-Type: image/jpeg` (for binary image upload), or standard JSON format.
- **JSON Request Payload:**
```json
{
  "image_b64": "base64_encoded_jpeg_string",
  "camera_id": "cam_entrance_1",
  "frame_id": "1711234567"
}
```
- **JSON Response:**
```json
{
  "success": true,
  "queue_length": 3,
  "average_wait_time": 45.2,
  "max_wait_time": 95.0,
  "camera_id": "cam_entrance_1",
  "frame_id": "1711234567",
  "persons": [
    {
      "x1": 120,
      "y1": 150,
      "x2": 240,
      "y2": 450,
      "confidence": 0.89,
      "track_id": 0,
      "wait_time_seconds": 95.0,
      "inside_roi": true
    }
  ],
  "processing_time_ms": 28.5
}
```
