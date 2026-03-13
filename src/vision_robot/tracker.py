# -*- coding: utf-8 -*-
"""ByteTrack ID 기반 경량 Kalman Filter 스무딩 모듈.

ByteTrack이 2D 이미지 공간에서 ID 추적을 담당하므로,
이 모듈은 ID별 3D 위치/속도 Kalman Filter 스무딩만 수행.
데이터 연관 없음 - 순수 스무딩 전용.
"""
import numpy as np

MAX_HUMAN_SPEED = 3.0  # m/s


class KFSmoother:
    """단일 트랙의 6-state KF 스무딩 (위치 + 속도)."""

    def __init__(self, position, stamp, q_pos=0.05, q_vel=0.5, r_pos=0.3):
        pos = np.array(position, dtype=np.float64)
        # state: [x, y, z, vx, vy, vz]
        self.x = np.array([pos[0], pos[1], pos[2], 0., 0., 0.], dtype=np.float64)
        # 공분산
        self.P = np.diag([r_pos, r_pos, r_pos, 1.0, 1.0, 1.0])
        # 프로세스 노이즈 파라미터
        self.q_pos = q_pos
        self.q_vel = q_vel
        # 측정 노이즈
        self.r_pos = r_pos
        # 측정 행렬 H: [I3x3, 0_3x3]
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        self.last_stamp = stamp
        self.lost_frames = 0
        self.hit_count = 1
        self.score = 0.0
        self.valid = True

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:]

    @property
    def distance(self):
        return float(np.linalg.norm(self.x[:2]))

    def predict(self, dt):
        """등속 모델 예측."""
        # F = [[I, dt*I], [0, I]]
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = F[1, 4] = F[2, 5] = dt

        self.x = F @ self.x

        # Q: 등속 프로세스 노이즈
        q = np.diag([
            self.q_pos, self.q_pos, self.q_pos,
            self.q_vel, self.q_vel, self.q_vel
        ]) * dt
        self.P = F @ self.P @ F.T + q

    def _update_kf(self, z):
        """KF 측정 업데이트."""
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + np.eye(3) * self.r_pos
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        # Joseph form
        self.P = I_KH @ self.P @ I_KH.T + K @ (np.eye(3) * self.r_pos) @ K.T

    def update(self, position, stamp, score=0.0):
        """새 관측으로 predict + update."""
        dt = (stamp - self.last_stamp).to_sec()
        if dt < 0.001:
            dt = 0.033  # ~30Hz fallback

        # predict
        self.predict(dt)

        z = np.array(position, dtype=np.float64)

        # 비현실적 점프 검사: innovation이 크면 R 증가
        innovation = np.linalg.norm(z[:2] - self.x[:2])
        if dt > 0 and innovation / dt > MAX_HUMAN_SPEED * 3:
            old_r = self.r_pos
            self.r_pos = 2.0  # 일시적으로 측정 불신
            self._update_kf(z)
            self.r_pos = old_r
        else:
            self._update_kf(z)

        # 속도 클램핑
        speed = np.linalg.norm(self.x[3:5])
        if speed > MAX_HUMAN_SPEED:
            self.x[3:5] *= MAX_HUMAN_SPEED / speed

        self.last_stamp = stamp
        self.score = score
        self.lost_frames = 0
        self.hit_count += 1
        self.valid = True

    def mark_lost(self, dt=0.033):
        """미탐지: predict만 수행 + 속도 감쇠."""
        self.predict(dt)
        self.x[3:] *= 0.7  # 속도 감쇠
        self.lost_frames += 1
        self.valid = False


class TrackedPersonStore:
    """ByteTrack ID 기반 KF 스무딩 저장소."""

    def __init__(self, max_lost_frames=5, q_pos=0.05, q_vel=0.5, r_pos=0.3):
        self.tracks = {}  # {track_id: KFSmoother}
        self.max_lost_frames = max_lost_frames
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos

    def update(self, track_id, position, stamp, score=0.0):
        """트랙 위치 업데이트 (KF 스무딩)."""
        if track_id in self.tracks:
            self.tracks[track_id].update(position, stamp, score)
        else:
            self.tracks[track_id] = KFSmoother(
                position, stamp, self.q_pos, self.q_vel, self.r_pos
            )
            self.tracks[track_id].score = score

    def get_prev_distance(self, track_id):
        """이전 프레임의 거리 반환 (서브 박스 계산용)."""
        if track_id in self.tracks:
            return self.tracks[track_id].distance
        return None

    def mark_lost_and_prune(self, active_ids):
        """활성 ID 외 트랙 lost 처리 및 오래된 트랙 제거."""
        to_remove = []
        for tid, smoother in self.tracks.items():
            if tid not in active_ids:
                smoother.mark_lost()
                if smoother.lost_frames > self.max_lost_frames:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def get_all_tracks(self):
        """모든 트랙 반환 (유효+lost 포함)."""
        return list(self.tracks.items())
