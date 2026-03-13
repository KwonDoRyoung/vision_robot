# -*- coding: utf-8 -*-
"""Kalman Filter 기반 다중 객체 트래킹 및 속도 추정 모듈.

상태 벡터: [x, y, z, vx, vy, vz]
관측 벡터: [x, y, z]
모델: 등속 운동 (constant velocity)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


# 사람의 최대 이동 속도 (m/s)
MAX_HUMAN_SPEED = 3.0


class KalmanFilter6D:
    """6차원 상태 (위치 3 + 속도 3) 선형 칼만 필터."""

    def __init__(self, process_noise_pos=0.05, process_noise_vel=0.5,
                 default_measurement_noise=0.3):
        """
        Args:
            process_noise_pos: 위치 프로세스 노이즈 표준편차 (m)
            process_noise_vel: 속도 프로세스 노이즈 표준편차 (m/s)
            default_measurement_noise: 기본 측정 노이즈 표준편차 (m)
        """
        self.n_state = 6
        self.n_obs = 3

        # 상태: [x, y, z, vx, vy, vz]
        self.x = np.zeros(self.n_state)
        # 공분산
        self.P = np.eye(self.n_state)
        self.P[0:3, 0:3] *= 1.0   # 초기 위치 불확실성
        self.P[3:6, 3:6] *= 10.0  # 초기 속도 불확실성 (모름)

        # 관측 행렬 H: 위치만 관측
        self.H = np.zeros((self.n_obs, self.n_state))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # 프로세스 노이즈 파라미터 (predict 시 dt에 따라 Q 생성)
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel

        # 기본 측정 노이즈
        self.default_r = default_measurement_noise

    def init_state(self, position):
        """초기 상태 설정."""
        self.x[:3] = position
        self.x[3:] = 0.0
        self.P = np.eye(self.n_state)
        self.P[0:3, 0:3] *= 0.5
        self.P[3:6, 3:6] *= 10.0

    def predict(self, dt):
        """등속 모델 예측 단계."""
        if dt < 1e-6:
            return

        # 상태 전이 행렬 F
        F = np.eye(self.n_state)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # 프로세스 노이즈 Q (연속시간 백색 노이즈 이산화)
        q_p = self.q_pos ** 2
        q_v = self.q_vel ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0

        Q = np.zeros((self.n_state, self.n_state))
        for i in range(3):
            Q[i, i] = q_p * dt + q_v * dt4       # pos-pos
            Q[i, i + 3] = q_v * dt3               # pos-vel
            Q[i + 3, i] = q_v * dt3               # vel-pos
            Q[i + 3, i + 3] = q_v * dt2           # vel-vel

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement, measurement_noise):
        """관측 업데이트 단계.

        Args:
            measurement: (3,) 관측 위치 [x, y, z]
            measurement_noise: 측정 노이즈 표준편차 (m)
                포인트 수 많을수록 작게, 적을수록 크게
        """
        z = np.asarray(measurement, dtype=np.float64)
        R = np.eye(self.n_obs) * (measurement_noise ** 2)

        # 잔차
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.n_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form (안정성)

        # 속도 클램핑
        speed = np.linalg.norm(self.x[3:5])
        if speed > MAX_HUMAN_SPEED:
            self.x[3:5] *= MAX_HUMAN_SPEED / speed

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()


class TrackedObject:
    _next_id = 1

    def __init__(self, position, score, stamp,
                 process_noise_pos=0.05, process_noise_vel=0.5):
        self.id = TrackedObject._next_id
        TrackedObject._next_id += 1

        self.kf = KalmanFilter6D(process_noise_pos, process_noise_vel)
        self.kf.init_state(position)

        self.score = score
        self.last_stamp = stamp
        self.lost_frames = 0
        self.hit_count = 1
        self.valid = True

    @property
    def position(self):
        return self.kf.position

    @property
    def velocity(self):
        return self.kf.velocity

    def predict(self, stamp):
        """칼만 예측 수행 및 예측 위치 반환."""
        dt = (stamp - self.last_stamp).to_sec()
        self.kf.predict(dt)
        return self.kf.position

    def update(self, position, score, stamp, measurement_noise=0.3):
        """칼만 업데이트 수행."""
        dt = (stamp - self.last_stamp).to_sec()

        # 물리적 점프 검사: 예측 위치와 관측의 차이
        new_pos = np.asarray(position, dtype=np.float64)
        predicted = self.kf.position
        displacement = np.linalg.norm(new_pos[:2] - predicted[:2])

        if dt > 1e-3 and displacement / dt > MAX_HUMAN_SPEED * 2:
            # 비현실적 점프 → 측정 노이즈를 크게 (관측 불신)
            measurement_noise = max(measurement_noise, 2.0)

        self.kf.update(new_pos, measurement_noise)

        self.score = score
        self.last_stamp = stamp
        self.lost_frames = 0
        self.hit_count += 1
        self.valid = True

    def mark_lost(self, stamp):
        """미탐지 시 predict만 수행 + 속도 감쇠."""
        dt = (stamp - self.last_stamp).to_sec()
        self.kf.predict(dt)
        # 속도 감쇠: lost 상태에서 드리프트 방지
        self.kf.x[3:6] *= 0.7
        self.last_stamp = stamp
        self.lost_frames += 1
        self.valid = False


def compute_measurement_noise(num_points, source="lidar",
                              min_points=20, max_points=50):
    """관측 포인트 수에 따라 측정 노이즈 결정.

    Args:
        num_points: bbox 내 유효 포인트 수
        source: "lidar" 또는 "depth"
        min_points: 최소 포인트 기준
        max_points: 최대 신뢰 포인트 기준

    Returns:
        measurement_noise: 측정 노이즈 표준편차 (m)
    """
    if source == "depth":
        # Depth는 LiDAR보다 기본 노이즈가 큼
        return 0.5

    if num_points >= max_points:
        return 0.1  # 최대 신뢰
    elif num_points >= min_points:
        # 선형 보간: 20개→0.4, 50개→0.1
        t = (num_points - min_points) / (max_points - min_points)
        return 0.4 - 0.3 * t
    else:
        # min_points 미만은 사용 안 함 (호출 전에 필터링)
        return 1.0


class PersonTracker:
    def __init__(self, max_distance=1.0, max_lost_frames=10,
                 process_noise_pos=0.05, process_noise_vel=0.5):
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.tracks = []

    def update(self, detections, stamp):
        """프레임 탐지 결과와 기존 트랙 매칭 및 갱신.

        Args:
            detections: list of (position, score, distance, measurement_noise)
                position: (3,) body 좌표
                score: 탐지 신뢰도
                distance: 로봇까지 거리
                measurement_noise: 측정 노이즈 (포인트 수 기반)
            stamp: rospy.Time

        Returns:
            list of TrackedObject (현재 활성 트랙)
        """
        if not self.tracks and not detections:
            return []

        # 탐지 없으면 모든 트랙 lost (predict만)
        if not detections:
            for t in self.tracks:
                t.mark_lost(stamp)
            self._prune()
            return self.tracks

        # 트랙 없으면 새로 생성
        if not self.tracks:
            for pos, score, _, _ in detections:
                self.tracks.append(TrackedObject(
                    pos, score, stamp,
                    self.process_noise_pos, self.process_noise_vel
                ))
            return self.tracks

        # 모든 트랙에 대해 predict 수행
        predicted_positions = []
        for t in self.tracks:
            pred = t.predict(stamp)
            predicted_positions.append(pred)

        # 예측 위치 기반 비용 행렬
        # lost 트랙은 매칭 거리를 확대 (놓쳤다 다시 잡기 위해)
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        cost = np.full((n_tracks, n_dets), 1e6)

        for i, pred in enumerate(predicted_positions):
            # lost_frames가 클수록 매칭 허용 거리 증가 (최대 2배)
            gate = self.max_distance * (1.0 + 0.2 * self.tracks[i].lost_frames)
            gate = min(gate, self.max_distance * 2.0)
            for j, (pos, _, _, _) in enumerate(detections):
                d = np.linalg.norm(pred[:2] - np.array(pos[:2]))
                if d < gate:
                    cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            gate = self.max_distance * (1.0 + 0.2 * self.tracks[r].lost_frames)
            gate = min(gate, self.max_distance * 2.0)
            if cost[r, c] < gate:
                pos, score, _, m_noise = detections[c]
                self.tracks[r].update(pos, score, stamp, m_noise)
                self.tracks[r].last_stamp = stamp
                matched_tracks.add(r)
                matched_dets.add(c)

        # 매칭 안 된 트랙 → lost (predict는 이미 수행됨)
        for i in range(n_tracks):
            if i not in matched_tracks:
                self.tracks[i].lost_frames += 1
                self.tracks[i].valid = False
                self.tracks[i].last_stamp = stamp

        # 매칭 안 된 탐지 → 새 트랙
        for j in range(n_dets):
            if j not in matched_dets:
                pos, score, _, _ = detections[j]
                self.tracks.append(TrackedObject(
                    pos, score, stamp,
                    self.process_noise_pos, self.process_noise_vel
                ))

        self._prune()
        return self.tracks

    def _prune(self):
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost_frames]
