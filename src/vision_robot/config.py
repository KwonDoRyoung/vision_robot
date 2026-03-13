# -*- coding: utf-8 -*-
import numpy as np
import rospy


def _load_matrix(param_name, default=None):
    flat = rospy.get_param(param_name, default)
    return np.array(flat, dtype=np.float64).reshape(4, 4)


def get_transforms():
    """body→sensor 변환 행렬 로드 및 파생 행렬 계산."""
    T_body_color = _load_matrix("~body_to_camera_color")
    T_body_depth = _load_matrix("~body_to_camera_depth")
    T_body_lidar = _load_matrix("~body_to_lidar")

    # 역행렬: sensor→body
    T_color_body = np.linalg.inv(T_body_color)
    T_depth_body = np.linalg.inv(T_body_depth)
    T_lidar_body = np.linalg.inv(T_body_lidar)

    # LiDAR→camera_color = (body→color) @ (lidar→body)
    T_lidar_to_color = T_body_color @ T_lidar_body

    return {
        "T_body_color": T_body_color,
        "T_body_depth": T_body_depth,
        "T_body_lidar": T_body_lidar,
        "T_color_body": T_color_body,
        "T_depth_body": T_depth_body,
        "T_lidar_body": T_lidar_body,
        "T_lidar_to_color": T_lidar_to_color,
    }


def get_color_intrinsics():
    vals = rospy.get_param("~color_intrinsics")
    fx, fy, cx, cy = vals
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K


def get_depth_intrinsics():
    vals = rospy.get_param("~depth_intrinsics")
    fx, fy, cx, cy = vals
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    return K
