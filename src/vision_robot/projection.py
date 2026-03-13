# -*- coding: utf-8 -*-
"""LiDAR→이미지 투영 및 Depth 기반 3D 위치 추정 모듈 (최적화 버전)."""
import numpy as np


def compute_sub_bbox(bbox, prev_distance=None,
                     close_ratio=0.35, far_ratio=0.55,
                     distance_threshold=2.0):
    """거리 기반 서브 바운딩 박스 계산.

    가까운 거리(0~2m): bbox가 크므로 배경 LiDAR 유입 → 작은 중심 영역
    먼 거리(2m+): bbox가 작으므로 더 넓은 중심 영역 사용

    Args:
        bbox: (x1, y1, x2, y2) 원본 바운딩 박스
        prev_distance: 이전 프레임 거리 (m), None이면 기본값 사용
        close_ratio: 가까운 거리에서 bbox 대비 서브 박스 비율
        far_ratio: 먼 거리에서 bbox 대비 서브 박스 비율
        distance_threshold: 가까운/먼 거리 기준 (m)

    Returns:
        (x1, y1, x2, y2) 서브 바운딩 박스
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    if prev_distance is not None and prev_distance < distance_threshold:
        # 가까운 거리: 작은 중심 영역 (배경 LiDAR 배제)
        ratio = close_ratio
    else:
        # 먼 거리 또는 첫 탐지: 넓은 중심 영역
        ratio = far_ratio

    sub_w = w * ratio
    sub_h = h * ratio
    # 약간 위쪽 (상체 중심, 다리/바닥 배제)
    cy_shift = -h * 0.05

    return (cx - sub_w * 0.5,
            cy + cy_shift - sub_h * 0.5,
            cx + sub_w * 0.5,
            cy + cy_shift + sub_h * 0.5)


def estimate_position_from_lidar(points_lidar, T_lidar_to_cam, T_lidar_body,
                                 K, bbox, img_w, img_h,
                                 sample_count=40, outlier_threshold=1.5,
                                 ground_height=0.2, roof_height=2.0,
                                 min_points=20):
    """LiDAR 포인트에서 bbox 내 객체의 3D 위치 추정 (최적화).

    Returns:
        position: (3,) body 좌표 또는 None
        distance: 거리 (m) 또는 None
        num_points: bbox 내 유효 포인트 수 (이상치 제거 후)
    """
    N = points_lidar.shape[0]
    if N == 0:
        return None, None, 0

    # 동차좌표 1번만 생성
    pts_h = np.empty((N, 4), dtype=np.float32)
    pts_h[:, :3] = points_lidar
    pts_h[:, 3] = 1.0

    # body 좌표 변환 (지면/천장 필터링용)
    R_lb = T_lidar_body[:3, :3].astype(np.float32)
    t_lb = T_lidar_body[:3, 3].astype(np.float32)
    pts_body = points_lidar @ R_lb.T + t_lb  # (N, 3) — 행렬곱 최적화

    # 지면/천장 필터링
    height_mask = (pts_body[:, 2] > ground_height) & (pts_body[:, 2] < roof_height)
    if not np.any(height_mask):
        return None, None, 0

    pts_h_f = pts_h[height_mask]
    pts_body_f = pts_body[height_mask]

    # 카메라 좌표 변환 + 이미지 투영 (한번에)
    R_lc = T_lidar_to_cam[:3, :3].astype(np.float32)
    t_lc = T_lidar_to_cam[:3, 3].astype(np.float32)
    pts_cam = pts_h_f[:, :3] @ R_lc.T + t_lc  # (M, 3)

    # 카메라 앞 (z > 0.1)
    front_mask = pts_cam[:, 2] > 0.1
    if not np.any(front_mask):
        return None, None, 0

    pts_cam = pts_cam[front_mask]
    pts_body_f = pts_body_f[front_mask]

    # 이미지 투영: uv = K @ pts_cam / z
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    inv_z = 1.0 / pts_cam[:, 2]
    u = pts_cam[:, 0] * inv_z * fx + cx
    v = pts_cam[:, 1] * inv_z * fy + cy

    # 이미지 범위 필터링
    x1, y1, x2, y2 = bbox
    bbox_mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    if not np.any(bbox_mask):
        return None, None, 0

    bbox_depths = pts_cam[bbox_mask, 2]
    bbox_body = pts_body_f[bbox_mask]
    num_points = len(bbox_depths)

    # 이상치 제거 (median 기반)
    if num_points > 2:
        med = np.median(bbox_depths)
        dev = np.abs(bbox_depths - med)
        mad = np.median(dev)
        if mad > 1e-6:
            inlier = dev / mad < outlier_threshold
            bbox_depths = bbox_depths[inlier]
            bbox_body = bbox_body[inlier]

    valid_count = len(bbox_body)
    if valid_count < min_points:
        return None, None, valid_count

    # 샘플링
    if valid_count > sample_count:
        idx = np.random.choice(valid_count, sample_count, replace=False)
        bbox_body = bbox_body[idx]

    # median 위치
    position = np.median(bbox_body, axis=0).astype(np.float64)
    distance = float(np.sqrt(position[0]**2 + position[1]**2))

    return position, distance, valid_count


def estimate_position_from_depth(depth_image, bbox, K_depth, T_depth_body,
                                 depth_scale=1000, min_depth=0.5, max_depth=6.0,
                                 margin=10):
    """Depth 이미지의 bbox 중심 영역에서 3D 위치 추정.

    Returns:
        position: (3,) body 좌표 또는 None
        distance: 거리 (m) 또는 None
    """
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h_d, w_d = depth_image.shape[:2]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = max((x2 - x1) // 4, margin)
    half_h = max((y2 - y1) // 4, margin)

    roi_x1 = max(0, cx - half_w)
    roi_y1 = max(0, cy - half_h)
    roi_x2 = min(w_d, cx + half_w)
    roi_y2 = min(h_d, cy + half_h)

    roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None, None

    roi_m = roi.astype(np.float32) / depth_scale
    valid_mask = (roi_m > min_depth) & (roi_m < max_depth)
    valid_depths = roi_m[valid_mask]

    if len(valid_depths) == 0:
        return None, None

    med_depth = float(np.median(valid_depths))

    fx, fy = K_depth[0, 0], K_depth[1, 1]
    cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]
    X_cam = (cx - cx_d) * med_depth / fx
    Y_cam = (cy - cy_d) * med_depth / fy
    Z_cam = med_depth

    pt_cam = np.array([X_cam, Y_cam, Z_cam, 1.0], dtype=np.float64)
    pt_body = T_depth_body @ pt_cam
    position = pt_body[:3]
    distance = float(np.sqrt(position[0]**2 + position[1]**2))

    return position, distance
