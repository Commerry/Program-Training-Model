"""
Run a model over an uploaded video and report what it found, frame by frame.

The result is a list of timestamps and boxes, not an annotated video file.
That is deliberate. Writing H.264 from OpenCV depends on an openh264 library
that is not reliably present — on the machine this was built on, the avc1
writer reported "Incorrect library version loaded" and produced a file no
browser would accept — so a server that returns a video works on one install
and silently produces something unplayable on the next.

Sending coordinates instead sidesteps the codec entirely: the browser already
has the video the user just chose, so it plays that locally and draws the boxes
over it. It is also far less data. The cost is that there is no annotated file
to download afterwards.

Inference runs in a worker thread rather than a subprocess: it holds a model
that the still-image and webcam paths also want cached, and a subprocess would
have to load its own copy.
"""

import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from services import inference
from services.projects import ProjectError

# A phone recording at 30 fps produces 1800 frames a minute, and each one costs
# roughly as much as testing a still image. Sampling keeps a long clip usable:
# detections are reported at the sampled instants and the page holds the last
# ones between samples, which is what a viewer sees anyway at these rates.
DEFAULT_SAMPLE_FPS = 5.0
MAX_SAMPLED_FRAMES = 3000
MAX_VIDEO_BYTES = 512 * 1024 * 1024
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mpg', '.mpeg'}

# Finished jobs are kept so the page can still read the result after the run
# ends, and dropped once there are too many to be anyone's current work.
MAX_KEPT_JOBS = 8

_lock = threading.Lock()
_jobs = {}


def _now():
    return datetime.now().isoformat()


def _public(job):
    """The parts of a job that are safe and useful to send to the browser."""
    return {
        'id': job['id'],
        'status': job['status'],
        'message': job['message'],
        'filename': job['filename'],
        'model_name': job['model_name'],
        'frames_total': job['frames_total'],
        'frames_done': job['frames_done'],
        'duration_s': job['duration_s'],
        'fps': job['fps'],
        'width': job['width'],
        'height': job['height'],
        'sample_fps': job['sample_fps'],
        'detection_count': job['detection_count'],
        'label_names': job['label_names'],
        'started_at': job['started_at'],
        'finished_at': job['finished_at'],
        'elapsed_s': job['elapsed_s'],
        # Only sent once the run is over; during the run this is thousands of
        # entries growing under the reader's feet.
        'frames': job['frames'] if job['status'] in ('completed', 'stopped') else [],
    }


def get(job_id):
    with _lock:
        job = _jobs.get(job_id)
        return _public(job) if job else None


def list_jobs():
    with _lock:
        return [_public(job) for job in
                sorted(_jobs.values(), key=lambda j: j['started_at'], reverse=True)]


def stop(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise ProjectError('No such video job', status=404)
        if job['status'] == 'running':
            job['cancel'] = True
            job['message'] = 'Stopping...'
        return _public(job)


def _prune():
    """Drop the oldest finished jobs, and their uploaded files with them."""
    finished = [j for j in _jobs.values() if j['status'] != 'running']
    for job in sorted(finished, key=lambda j: j['started_at'])[:-MAX_KEPT_JOBS or None]:
        _discard(job)


def _discard(job):
    _jobs.pop(job['id'], None)
    path = job.get('source_path')
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            # On Windows the decoder can still hold the handle briefly; leaving
            # a temporary file behind is better than failing the request.
            pass


def start(model_path, video_file, score_threshold=0.5, label_names=None,
          img_size=640, sample_fps=DEFAULT_SAMPLE_FPS, work_dir=None):
    """
    Save the upload and begin analysing it. Returns the job immediately.
    """
    if video_file is None or not getattr(video_file, 'filename', ''):
        raise ProjectError('Select a video file')

    suffix = Path(video_file.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ProjectError(f'Supported video files: {", ".join(sorted(VIDEO_SUFFIXES))}')

    try:
        sample_fps = min(max(float(sample_fps), 0.2), 30.0)
    except (TypeError, ValueError):
        sample_fps = DEFAULT_SAMPLE_FPS

    work_dir = Path(work_dir) if work_dir else Path(_default_work_dir())
    work_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    source_path = work_dir / f'{job_id}{suffix}'
    video_file.save(str(source_path))

    size = source_path.stat().st_size
    if size > MAX_VIDEO_BYTES:
        source_path.unlink(missing_ok=True)
        raise ProjectError(
            f'That video is {size / 1e6:.0f} MB, over the '
            f'{MAX_VIDEO_BYTES / 1e6:.0f} MB limit.')

    probe = cv2.VideoCapture(str(source_path))
    if not probe.isOpened():
        probe.release()
        source_path.unlink(missing_ok=True)
        raise ProjectError(
            'That file could not be opened as a video. Some phone recordings '
            'use a codec OpenCV was not built with; converting it to H.264 mp4 '
            'usually fixes it.')
    fps = float(probe.get(cv2.CAP_PROP_FPS) or 0)
    frames_total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()

    if fps <= 0:
        fps = 30.0            # some containers do not record it
    duration = frames_total / fps if frames_total else 0.0

    job = {
        'id': job_id,
        'status': 'running',
        'message': 'Reading the video...',
        'filename': Path(video_file.filename).name,
        'model_name': Path(model_path).name,
        'model_path': str(model_path),
        'source_path': str(source_path),
        'score_threshold': score_threshold,
        'label_names': list(label_names or []),
        'img_size': img_size,
        'sample_fps': sample_fps,
        'fps': round(fps, 3),
        'width': width,
        'height': height,
        'duration_s': round(duration, 2),
        'frames_total': 0,
        'frames_done': 0,
        'detection_count': 0,
        'frames': [],
        'started_at': _now(),
        'finished_at': None,
        'elapsed_s': 0.0,
        'cancel': False,
    }

    with _lock:
        _prune()
        _jobs[job_id] = job

    thread = threading.Thread(target=_run, args=(job_id,), daemon=True)
    thread.start()
    return _public(job)


def _default_work_dir():
    from config import INSTANCE_DIR
    return Path(INSTANCE_DIR) / 'video-tests'


def _run(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return

    started = time.time()
    capture = cv2.VideoCapture(job['source_path'])
    try:
        fps = job['fps']
        stride = max(1, int(round(fps / job['sample_fps'])))
        index = 0
        sampled = 0
        total_detections = 0
        frames = []
        labels_in_use = job['label_names']

        while True:
            if job['cancel']:
                job['status'] = 'stopped'
                job['message'] = f'Stopped after {sampled} sampled frames.'
                break

            ok, frame = capture.read()
            if not ok:
                job['status'] = 'completed'
                job['message'] = (f'{sampled} frames sampled, '
                                  f'{total_detections} detections.')
                break

            if index % stride == 0:
                detections, labels_in_use = inference.detect_frame(
                    job['model_path'], frame,
                    score_threshold=job['score_threshold'],
                    label_names=job['label_names'],
                    img_size=job['img_size'])
                total_detections += len(detections)
                frames.append({
                    'time_s': round(index / fps, 3),
                    'frame': index,
                    'detections': detections,
                    'reading': inference.reading_of(detections),
                })
                sampled += 1
                job['frames_done'] = sampled
                job['detection_count'] = total_detections
                job['elapsed_s'] = round(time.time() - started, 1)

                if sampled >= MAX_SAMPLED_FRAMES:
                    job['status'] = 'completed'
                    job['message'] = (
                        f'Stopped at the {MAX_SAMPLED_FRAMES}-frame limit '
                        f'({round(index / fps, 1)}s of {job["duration_s"]}s). '
                        'Lower the sample rate to cover the whole clip.')
                    break

            index += 1

        job['frames'] = frames
        job['frames_total'] = sampled
        job['label_names'] = labels_in_use
    except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
        job['status'] = 'failed'
        job['message'] = f'{type(exc).__name__}: {exc}'
    finally:
        capture.release()
        job['elapsed_s'] = round(time.time() - started, 1)
        job['finished_at'] = _now()
        # The upload is not needed once the boxes are known; the browser plays
        # its own copy.
        try:
            Path(job['source_path']).unlink(missing_ok=True)
        except OSError:
            pass
