"""
Violation Clip Recorder Module
Records and exports high-definition video clips of detected violations,
including pre-roll and post-roll temporal context with alert overlays.
Uses H.264 (libx264) encoding for native HTML5 web browser playback.
"""

from collections import deque
import os
import threading
from typing import Dict, List, Optional
import cv2
import numpy as np
from violation_detector import ViolationEvent

class ActiveClipSession:
    def __init__(self, event: ViolationEvent, target_path: str, max_post_roll: int = 45):
        self.event = event
        self.target_path = target_path
        self.max_post_roll = max_post_roll
        self.frames: List[np.ndarray] = []
        self.is_recording = True
        self.remaining_post_roll = max_post_roll

class ViolationClipRecorder:
    def __init__(
        self,
        output_dir: str,
        fps: float = 30.0,
        pre_roll_frames: int = 45,
        post_roll_frames: int = 45
    ):
        self.output_dir = output_dir
        self.fps = max(1.0, fps)
        self.pre_roll_frames = pre_roll_frames
        self.post_roll_frames = post_roll_frames
        
        # Rolling buffer of recent frames: deque of (frame_idx, frame_image)
        self.frame_buffer: deque = deque(maxlen=self.pre_roll_frames)
        
        # Active clip recording sessions: violation_id -> ActiveClipSession
        self.active_sessions: Dict[str, ActiveClipSession] = {}
        
        os.makedirs(self.output_dir, exist_ok=True)

    def buffer_frame(self, frame_idx: int, frame: np.ndarray):
        """Adds frame to pre-roll ring buffer and feeds active recording sessions."""
        self.frame_buffer.append((frame_idx, frame.copy()))

        finished_ids = []
        for vid, session in self.active_sessions.items():
            if session.is_recording:
                session.frames.append(frame.copy())
                if not session.event.is_active:
                    session.remaining_post_roll -= 1
                    if session.remaining_post_roll <= 0:
                        session.is_recording = False
                        finished_ids.append(vid)

        for vid in finished_ids:
            self._save_session_clip(vid)

    def trigger_violation_clip(self, event: ViolationEvent):
        """
        Initiates a new clip recording session for confirmed violation.
        Pre-populates it with frames from pre-roll buffer.
        """
        safe_type = event.violation_type.value.split()[0].lower()
        filename = f"clip_{event.violation_id}_{safe_type}_veh{event.vehicle_id}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        session = ActiveClipSession(
            event=event,
            target_path=filepath,
            max_post_roll=self.post_roll_frames
        )

        # Prepend frames from pre-roll buffer
        for _, buf_frame in self.frame_buffer:
            session.frames.append(buf_frame.copy())

        event.clip_path = filepath
        self.active_sessions[event.violation_id] = session

    def _save_session_clip(self, violation_id: str):
        session = self.active_sessions.pop(violation_id, None)
        if not session or len(session.frames) == 0:
            return

        frames_to_write = list(session.frames)
        session.event.clip_path = session.target_path

        # Write asynchronously in a background thread so the video processing thread never lags
        threading.Thread(
            target=self._write_clip_thread,
            args=(session.target_path, frames_to_write),
            daemon=True
        ).start()

    def _write_clip_thread(self, target_path: str, frames: List[np.ndarray]):
        if not frames:
            return

        h, w = frames[0].shape[:2]
        
        # Use H.264 libx264 encoding with faststart for native in-browser HTML5 playback
        saved_h264 = False
        try:
            import imageio_ffmpeg
            writer = imageio_ffmpeg.write_frames(
                target_path,
                (w, h),
                fps=self.fps,
                codec='libx264',
                pix_fmt_in='bgr24',
                pix_fmt_out='yuv420p',
                output_params=['-movflags', '+faststart'],
                macro_block_size=1
            )
            writer.send(None)
            for f in frames:
                writer.send(f)
            writer.close()
            saved_h264 = True
        except Exception:
            saved_h264 = False

        # Fallback to OpenCV mp4v if H.264 encoder failed
        if not saved_h264:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(target_path, fourcc, self.fps, (w, h))
            for f in frames:
                out.write(f)
            out.release()

    def finalize_all(self):
        """Saves any remaining in-progress clips upon video completion."""
        for vid in list(self.active_sessions.keys()):
            self._save_session_clip(vid)
