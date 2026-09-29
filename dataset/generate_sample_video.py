"""
Synthetic Crowd Video Generator for Normal & Multiple Anomalous Crowd Dynamics Scenarios.

Includes scenarios:
1. Normal Crowd Flow (data/sample_crowd_normal.mp4)
2. Panic Scatter & Counter-Flow (data/sample_crowd_test.mp4)
3. Crowd Stampede & Bottleneck Blockade (data/scenario_stampede.mp4)
4. Multi-Zone Concurrent Disturbances (data/scenario_multizone.mp4)
5. Night Vision Surveillance Crowd Surge (data/scenario_night_surge.mp4)
"""

import os
import cv2
import numpy as np


class SyntheticCrowdGenerator:
    def __init__(self, width=320, height=240, num_pedestrians=50, fps=20):
        self.width = width
        self.height = height
        self.num_pedestrians = num_pedestrians
        self.fps = fps

    def _create_background(self, style="day"):
        if style == "day":
            bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40
            cv2.line(bg, (0, 30), (self.width, 30), (70, 70, 70), 2)
            cv2.line(bg, (0, self.height - 30), (self.width, self.height - 30), (70, 70, 70), 2)
            for x in range(0, self.width, 30):
                cv2.line(bg, (x, self.height // 2), (x + 15, self.height // 2), (60, 60, 60), 1)
            return bg
        elif style == "night":
            # Night vision background (dark green hue)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            bg[:, :, 1] = 25  # subtle green tint
            bg[:, :, 0] = 5
            bg[:, :, 2] = 5
            # Add grid crosshairs
            cv2.line(bg, (self.width // 2, 0), (self.width // 2, self.height), (0, 50, 0), 1)
            cv2.line(bg, (0, self.height // 2), (self.width, self.height // 2), (0, 50, 0), 1)
            return bg

    def generate_normal_video(self, output_path, total_frames=160):
        """Generates video containing ONLY normal steady crowd dynamics."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(42)
        positions = np.random.rand(self.num_pedestrians, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 40 + positions[:, 1] * (self.height - 80)
        
        velocities = np.zeros((self.num_pedestrians, 2))
        velocities[:, 0] = np.random.uniform(1.5, 3.0, size=self.num_pedestrians)
        velocities[:, 1] = np.random.uniform(-0.3, 0.3, size=self.num_pedestrians)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            positions += velocities
            
            out_of_bounds = positions[:, 0] > self.width
            positions[out_of_bounds, 0] = 0
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(self.num_pedestrians):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)
                cv2.circle(frame, (px, py), 5, (220, 240, 150), 1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved normal crowd video to: {output_path} ({total_frames} frames)")

    def generate_anomaly_video(self, output_path, total_frames=240):
        """Generates baseline anomaly video (Panic Scatter & Counter-Flow Rush)."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(100)
        positions = np.random.rand(self.num_pedestrians, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 40 + positions[:, 1] * (self.height - 80)

        velocities = np.zeros((self.num_pedestrians, 2))
        velocities[:, 0] = np.random.uniform(1.5, 3.0, size=self.num_pedestrians)
        velocities[:, 1] = np.random.uniform(-0.3, 0.3, size=self.num_pedestrians)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()

            is_anomaly_1 = (70 <= frame_idx <= 120)   # Panic scatter in center-right
            is_anomaly_2 = (170 <= frame_idx <= 210)  # Counter-flow rush in top-left

            curr_vel = velocities.copy()

            if is_anomaly_1:
                mask_group = (positions[:, 0] > 160) & (positions[:, 0] < 270) & (positions[:, 1] > 60) & (positions[:, 1] < 180)
                angles = np.random.uniform(0, 2 * np.pi, size=self.num_pedestrians)
                curr_vel[mask_group, 0] = np.cos(angles[mask_group]) * np.random.uniform(8.0, 14.0, size=mask_group.sum())
                curr_vel[mask_group, 1] = np.sin(angles[mask_group]) * np.random.uniform(8.0, 14.0, size=mask_group.sum())

            elif is_anomaly_2:
                mask_group = (positions[:, 0] > 30) & (positions[:, 0] < 150) & (positions[:, 1] > 30) & (positions[:, 1] < 120)
                curr_vel[mask_group, 0] = np.random.uniform(-10.0, -6.0, size=mask_group.sum())
                curr_vel[mask_group, 1] = np.random.uniform(-1.5, 1.5, size=mask_group.sum())

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 40
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(self.num_pedestrians):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)
                cv2.circle(frame, (px, py), 5, (220, 240, 150), 1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved anomaly test crowd video to: {output_path} ({total_frames} frames)")

    def generate_stampede_video(self, output_path, total_frames=200):
        """Scenario 1: High-density crowd stampede & bottleneck blockade."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(200)
        n_ped = 70
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(2.0, 3.5, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.2, 0.2, size=n_ped)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Bottleneck blockade active (Frames 60..130) at X: 220, Y: 120
            is_stampede = (60 <= frame_idx <= 130)

            if is_stampede:
                # Draw red barrier icon at bottleneck location
                cv2.rectangle(frame, (215, 80), (225, 160), (0, 0, 255), -1)
                cv2.putText(frame, "BARRIER", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                # Crowd near barrier halts & crushes into panic surge
                near_barrier = (positions[:, 0] > 170) & (positions[:, 0] < 235) & (positions[:, 1] > 60) & (positions[:, 1] < 180)
                curr_vel[near_barrier, 0] = np.random.uniform(-6.0, 2.0, size=near_barrier.sum())
                curr_vel[near_barrier, 1] = np.random.uniform(-8.0, 8.0, size=near_barrier.sum())

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved stampede scenario video to: {output_path} ({total_frames} frames)")

    def generate_multizone_video(self, output_path, total_frames=220):
        """Scenario 2: Multi-Zone Concurrent Disturbances."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(300)
        n_ped = 75
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(1.5, 3.0, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.4, 0.4, size=n_ped)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Event A: High-speed intruder/run in bottom-left (Frames 45..95)
            if 45 <= frame_idx <= 95:
                grp_a = (positions[:, 0] > 20) & (positions[:, 0] < 120) & (positions[:, 1] > 130) & (positions[:, 1] < 210)
                curr_vel[grp_a, 0] = np.random.uniform(9.0, 15.0, size=grp_a.sum())
                curr_vel[grp_a, 1] = np.random.uniform(-3.0, 3.0, size=grp_a.sum())

            # Event B: Rapid circular gathering / fight in top-right (Frames 125..175)
            if 125 <= frame_idx <= 175:
                grp_b = (positions[:, 0] > 180) & (positions[:, 0] < 290) & (positions[:, 1] > 30) & (positions[:, 1] < 110)
                # Velocities pull inward toward center (235, 70)
                dx = 235 - positions[grp_b, 0]
                dy = 70 - positions[grp_b, 1]
                dist = np.sqrt(dx**2 + dy**2) + 1e-5
                curr_vel[grp_b, 0] = (dx / dist) * 7.0
                curr_vel[grp_b, 1] = (dy / dist) * 7.0

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved multi-zone scenario video to: {output_path} ({total_frames} frames)")

    def generate_night_surge_video(self, output_path, total_frames=200):
        """Scenario 3: Night Vision Surveillance Crowd Surge."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(400)
        n_ped = 60
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(1.2, 2.5, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.2, 0.2, size=n_ped)

        bg = self._create_background("night")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Night surge intrusion (Frames 55..135)
            if 55 <= frame_idx <= 135:
                grp_surge = (positions[:, 0] > 60) & (positions[:, 0] < 220) & (positions[:, 1] > 60) & (positions[:, 1] < 180)
                curr_vel[grp_surge, 0] = np.random.uniform(8.0, 13.0, size=grp_surge.sum())
                curr_vel[grp_surge, 1] = np.random.uniform(-4.0, 4.0, size=grp_surge.sum())

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                # Green night vision thermal glow
                cv2.circle(frame, (px, py), 5, (0, 255, 120), -1)
                cv2.circle(frame, (px, py), 7, (0, 200, 80), 1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved night surge scenario video to: {output_path} ({total_frames} frames)")

    def generate_evacuation_video(self, output_path, total_frames=200):
        """Scenario 5: Fire Alarm Evacuation Rush."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(500)
        n_ped = 65
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(1.5, 2.5, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.2, 0.2, size=n_ped)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Evacuation alarm active (Frames 50..140): All pedestrians rush toward right exit gate (310, 120)
            is_evac = (50 <= frame_idx <= 140)

            if is_evac:
                cv2.rectangle(frame, (310, 80), (320, 160), (0, 255, 0), -1)
                cv2.putText(frame, "EXIT", (270, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                dx = 310 - positions[:, 0]
                dy = 120 - positions[:, 1]
                dist = np.sqrt(dx**2 + dy**2) + 1e-5
                curr_vel[:, 0] = (dx / dist) * np.random.uniform(8.0, 12.0, size=n_ped)
                curr_vel[:, 1] = (dy / dist) * np.random.uniform(8.0, 12.0, size=n_ped)

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved evacuation scenario video to: {output_path} ({total_frames} frames)")

    def generate_intrusion_video(self, output_path, total_frames=200):
        """Scenario 6: High-Speed Vehicle Intrusion in Pedestrian Plaza."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(600)
        n_ped = 55
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(1.5, 2.8, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.3, 0.3, size=n_ped)

        bg = self._create_background("day")

        # Intruder object parameters
        veh_y = 0.0

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Vehicle intrusion active (Frames 60..130)
            is_intrusion = (60 <= frame_idx <= 130)

            if is_intrusion:
                veh_y += 5.0
                vx, vy = 160, int(veh_y)
                cv2.rectangle(frame, (vx - 12, vy - 20), (vx + 12, vy + 20), (0, 0, 255), -1)
                cv2.putText(frame, "INTRUDER", (vx - 25, max(vy - 25, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                # Pedestrians scatter radially away from vehicle
                dx = positions[:, 0] - vx
                dy = positions[:, 1] - vy
                dist = np.sqrt(dx**2 + dy**2) + 1e-5
                close_mask = dist < 70
                curr_vel[close_mask, 0] = (dx[close_mask] / dist[close_mask]) * 10.0
                curr_vel[close_mask, 1] = (dy[close_mask] / dist[close_mask]) * 10.0

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved vehicle intrusion scenario video to: {output_path} ({total_frames} frames)")

    def generate_freeze_video(self, output_path, total_frames=200):
        """Scenario 7: Sudden Crowd Freeze / Static Obstacle Anomaly."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        np.random.seed(700)
        n_ped = 50
        positions = np.random.rand(n_ped, 2)
        positions[:, 0] *= self.width
        positions[:, 1] = 30 + positions[:, 1] * (self.height - 60)

        velocities = np.zeros((n_ped, 2))
        velocities[:, 0] = np.random.uniform(1.8, 3.0, size=n_ped)
        velocities[:, 1] = np.random.uniform(-0.3, 0.3, size=n_ped)

        bg = self._create_background("day")

        for frame_idx in range(total_frames):
            frame = bg.copy()
            curr_vel = velocities.copy()

            # Freeze anomaly active (Frames 60..130): Center group halts dead in place
            if 60 <= frame_idx <= 130:
                freeze_mask = (positions[:, 0] > 110) & (positions[:, 0] < 210) & (positions[:, 1] > 70) & (positions[:, 1] < 170)
                curr_vel[freeze_mask, 0] = 0.0
                curr_vel[freeze_mask, 1] = 0.0

            positions += curr_vel

            out_of_bounds = (positions[:, 0] > self.width) | (positions[:, 0] < 0) | (positions[:, 1] < 20) | (positions[:, 1] > self.height - 20)
            positions[out_of_bounds, 0] = np.random.rand(out_of_bounds.sum()) * 30
            positions[out_of_bounds, 1] = 40 + np.random.rand(out_of_bounds.sum()) * (self.height - 80)

            for i in range(n_ped):
                px, py = int(positions[i, 0]), int(positions[i, 1])
                cv2.circle(frame, (px, py), 4, (180, 200, 100), -1)

            out.write(frame)

        out.release()
        print(f"[Generator] Saved crowd freeze scenario video to: {output_path} ({total_frames} frames)")


if __name__ == "__main__":
    gen = SyntheticCrowdGenerator()
    gen.generate_normal_video("data/sample_crowd_normal.mp4", total_frames=160)
    gen.generate_anomaly_video("data/sample_crowd_test.mp4", total_frames=240)
    gen.generate_stampede_video("data/scenario_stampede.mp4", total_frames=200)
    gen.generate_multizone_video("data/scenario_multizone.mp4", total_frames=220)
    gen.generate_night_surge_video("data/scenario_night_surge.mp4", total_frames=200)
    gen.generate_evacuation_video("data/scenario_fire_evacuation.mp4", total_frames=200)
    gen.generate_intrusion_video("data/scenario_vehicle_intrusion.mp4", total_frames=200)
    gen.generate_freeze_video("data/scenario_crowd_freeze.mp4", total_frames=200)
