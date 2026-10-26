"""
Synthetic Traffic Video Generator
Creates a realistic-looking overhead traffic scene with:
 - Multiple normally-flowing vehicles in each lane
 - One deliberate WRONG-WAY vehicle (going opposite to lane flow)
 - One deliberate LANE-STRADDLING vehicle
Used as a fallback when no real video is provided.
"""

import math
import random
import cv2
import numpy as np

# --------------------------------------------------------------------------
# Helper: draw a coloured vehicle rectangle with a small direction arrow
# --------------------------------------------------------------------------

def draw_vehicle(frame, x, y, w, h, color, heading_deg=0.0):
    x1, y1 = int(x - w / 2), int(y - h / 2)
    x2, y2 = int(x + w / 2), int(y + h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    # windshield
    cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y1 + int(h * 0.3)), (20, 20, 20), -1)
    # small heading arrow
    cx, cy = int(x), int(y)
    rad = math.radians(heading_deg)
    ax = int(cx + 20 * math.cos(rad))
    ay = int(cy + 20 * math.sin(rad))
    cv2.arrowedLine(frame, (cx, cy), (ax, ay), (255, 255, 0), 2, tipLength=0.4)


def draw_road(frame, w, h):
    """Draw a 4-lane divided highway with dashed lane markings."""
    frame[:] = (55, 55, 55)  # Dark asphalt

    # Lane boundaries (y)
    lane_ys = [0, int(h * 0.25), int(h * 0.5), int(h * 0.75), h]
    # Yellow centre divider
    cy = int(h * 0.5)
    cv2.rectangle(frame, (0, cy - 3), (w, cy + 3), (0, 200, 200), -1)

    # White lane markings (dashed)
    for lane_boundary_y in [int(h * 0.25), int(h * 0.75)]:
        for dash_x in range(0, w, 60):
            cv2.rectangle(frame, (dash_x, lane_boundary_y - 2),
                          (dash_x + 35, lane_boundary_y + 2), (200, 200, 200), -1)

    # Solid outer kerbs
    cv2.rectangle(frame, (0, 0), (w, 4), (200, 200, 200), -1)
    cv2.rectangle(frame, (0, h - 4), (w, h), (200, 200, 200), -1)

    # Direction arrows per lane strip
    arrow_xs = [w // 4, 3 * w // 4]
    for ax in arrow_xs:
        # EB arrows (top half)
        for ly in [int(h * 0.125), int(h * 0.375)]:
            cv2.arrowedLine(frame, (ax - 30, ly), (ax + 30, ly),
                            (120, 120, 120), 2, tipLength=0.25)
        # WB arrows (bottom half)
        for ly in [int(h * 0.625), int(h * 0.875)]:
            cv2.arrowedLine(frame, (ax + 30, ly), (ax - 30, ly),
                            (120, 120, 120), 2, tipLength=0.25)


class SyntheticVehicle:
    def __init__(self, x, y, vx, vy, color, vtype="car", w=52, h=28):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.vtype = vtype
        self.w = w
        self.h = h

    @property
    def heading(self):
        return math.degrees(math.atan2(self.vy, self.vx)) % 360.0

    def step(self):
        self.x += self.vx
        self.y += self.vy

    def is_out(self, W, H):
        return self.x < -200 or self.x > W + 200 or self.y < -100 or self.y > H + 100


def generate_sample_traffic_video(
    output_path: str = "sample_traffic.mp4",
    duration_seconds: int = 20,
    fps: int = 25,
    width: int = 1280,
    height: int = 720
):
    total_frames = duration_seconds * fps
    h, w = height, width
    lane_centers = [
        int(h * 0.125),   # EB fast
        int(h * 0.375),   # EB slow
        int(h * 0.625),   # WB fast
        int(h * 0.875),   # WB slow
    ]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ---- spawn normal fleet ------------------------------------------------
    rng = random.Random(42)
    vehicles = []
    # EB lanes: x starts negative, moving rightward
    for lane_y in lane_centers[:2]:
        for _ in range(6):
            sx = rng.randint(-600, -60)
            speed = rng.uniform(6, 11)
            color = (rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200))
            vehicles.append(SyntheticVehicle(sx, lane_y + rng.randint(-20, 20), speed, 0, color))
    # WB lanes: x starts at far right, moving leftward
    for lane_y in lane_centers[2:]:
        for _ in range(6):
            sx = rng.randint(w + 60, w + 600)
            speed = rng.uniform(6, 10)
            color = (rng.randint(40, 200), rng.randint(40, 200), rng.randint(40, 200))
            vehicles.append(SyntheticVehicle(sx, lane_y + rng.randint(-20, 20), -speed, 0, color))

    # ---- WRONG-WAY vehicle: starts on EB lane but moves LEFTWARD -----------
    wrong_way = SyntheticVehicle(
        x=w - 50,
        y=lane_centers[0],   # Top (EB) lane
        vx=-9.0,              # Moving left = WRONG WAY on EB lane
        vy=0.0,
        color=(0, 0, 220),    # Red/blue to make it obvious visually
        vtype="car"
    )

    # ---- STRADDLING vehicle: rides the centre of EB lane boundary ----------
    straddle_y = (lane_centers[0] + lane_centers[1]) / 2.0  # on the lane line
    straddler = SyntheticVehicle(
        x=-80,
        y=straddle_y,
        vx=7.5,
        vy=0.0,
        color=(255, 120, 0),
        vtype="car"
    )

    spawn_timer = 0
    for frame_num in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        draw_road(frame, w, h)

        # Step & respawn normal vehicles
        for v in vehicles:
            v.step()
        # Respawn off-screen EB vehicles
        for v in vehicles[:12]:
            if v.is_out(w, h):
                lane_y = rng.choice(lane_centers[:2])
                v.x = rng.randint(-400, -60)
                v.y = lane_y + rng.randint(-20, 20)
                v.vx = rng.uniform(6, 11)
        # Respawn off-screen WB vehicles
        for v in vehicles[12:]:
            if v.is_out(w, h):
                lane_y = rng.choice(lane_centers[2:])
                v.x = rng.randint(w + 60, w + 400)
                v.y = lane_y + rng.randint(-20, 20)
                v.vx = -rng.uniform(6, 10)

        # Wrong-way vehicle: wrap around
        wrong_way.step()
        if wrong_way.x < -120:
            wrong_way.x = w + 40

        # Straddler: wrap around
        straddler.step()
        if straddler.x > w + 120:
            straddler.x = -80

        # Draw all vehicles
        for v in vehicles:
            draw_vehicle(frame, v.x, v.y, v.w, v.h, v.color, v.heading)

        # Draw wrong-way vehicle with red border highlight
        draw_vehicle(frame, wrong_way.x, wrong_way.y, 52, 28, (30, 30, 200), wrong_way.heading)
        cv2.rectangle(frame,
                      (int(wrong_way.x - 28), int(wrong_way.y - 16)),
                      (int(wrong_way.x + 28), int(wrong_way.y + 16)),
                      (0, 0, 255), 2)

        draw_vehicle(frame, straddler.x, straddler.y, 52, 38, (30, 120, 255), straddler.heading)

        # HUD
        cv2.putText(frame, "SYNTHETIC TRAFFIC DEMO | AI MONITORING",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        out.write(frame)

    out.release()
    print(f"[*] Synthetic traffic video saved to: {output_path} ({total_frames} frames, {fps} FPS)")


if __name__ == "__main__":
    generate_sample_traffic_video("sample_traffic_synthetic.mp4", duration_seconds=20, fps=25)
