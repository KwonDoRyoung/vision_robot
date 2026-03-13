#!/home/irop/projects/vision_robot/venv/bin/python3
# -*- coding: utf-8 -*-
"""Phase 3: Kalman Filter 기반 LiDAR/Depth 융합 사람 위치·속도 추정 노드.

Subscribe:
    /detection/persons (vision_msgs/Detection2DArray)
    /camera/depth/image_rect_raw (sensor_msgs/Image)
    /rslidar_points (sensor_msgs/PointCloud2)

Publish:
    /tracked_persons (vision_robot/TrackedPersonArray)
    /tracked_persons/markers (visualization_msgs/MarkerArray)
"""
import cv2  # cv_bridge보다 먼저 import 필요
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

from vision_robot.msg import TrackedPerson, TrackedPersonArray
from vision_robot.config import get_transforms, get_color_intrinsics, get_depth_intrinsics
from vision_robot.projection import (
    estimate_position_from_depth,
    estimate_position_from_lidar,
)
from vision_robot.tracker import PersonTracker, compute_measurement_noise


class PersonTrackerNode:
    def __init__(self):
        rospy.init_node("person_tracker_node", anonymous=False)

        # 센서 토픽
        depth_topic = rospy.get_param("~camera_depth_topic", "/camera/depth/image_rect_raw")
        lidar_topic = rospy.get_param("~lidar_topic", "/rslidar_points")
        det_topic = rospy.get_param("~detection_topic", "/detection/persons")

        # Depth 파라미터
        self.depth_scale = rospy.get_param("~depth_scale_factor", 1000)
        self.depth_min = rospy.get_param("~depth/min_value", 0.5)
        self.depth_max = rospy.get_param("~depth/max_value", 6.0)
        self.depth_margin = rospy.get_param("~depth/roi_margin", 10)
        self.img_w = rospy.get_param("~image_cols", 848)
        self.img_h = rospy.get_param("~image_rows", 480)

        # LiDAR 융합 파라미터
        self.sample_count = rospy.get_param("~fusion/sample_count", 40)
        self.outlier_thresh = rospy.get_param("~fusion/outlier_threshold", 1.5)
        self.ground_height = rospy.get_param("~fusion/ground_height", 0.2)
        self.roof_height = rospy.get_param("~fusion/roof_height", 2.0)
        self.min_lidar_points = rospy.get_param("~fusion/min_lidar_points", 20)
        self.max_lidar_points = rospy.get_param("~fusion/max_lidar_points", 50)

        # 트래커 파라미터
        max_dist = rospy.get_param("~tracker/max_distance", 1.0)
        max_lost = rospy.get_param("~tracker/max_lost_frames", 10)
        process_noise_pos = rospy.get_param("~tracker/process_noise_pos", 0.05)
        process_noise_vel = rospy.get_param("~tracker/process_noise_vel", 0.5)

        # 변환 행렬 / intrinsics
        self.transforms = get_transforms()
        self.K_color = get_color_intrinsics()
        self.K_depth = get_depth_intrinsics()

        # 트래커 (Kalman Filter 기반)
        self.tracker = PersonTracker(max_dist, max_lost,
                                     process_noise_pos, process_noise_vel)

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_lidar = None

        # Publisher
        self.pub = rospy.Publisher("/tracked_persons", TrackedPersonArray, queue_size=10)
        self.marker_pub = rospy.Publisher("/tracked_persons/markers", MarkerArray, queue_size=1)

        # Subscribers
        rospy.Subscriber(depth_topic, Image, self._depth_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(lidar_topic, PointCloud2, self._lidar_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(det_topic, Detection2DArray, self._det_cb, queue_size=1)

        rospy.loginfo("PersonTrackerNode 시작 (Kalman Filter)")
        rospy.spin()

    def _depth_cb(self, msg):
        self.latest_depth = msg

    def _lidar_cb(self, msg):
        self.latest_lidar = msg

    @staticmethod
    def _parse_pointcloud(msg):
        """PointCloud2를 numpy 배열로 직접 파싱 (pc2.read_points 대비 10x+ 빠름)."""
        x_off = y_off = z_off = None
        for f in msg.fields:
            if f.name == 'x': x_off = f.offset
            elif f.name == 'y': y_off = f.offset
            elif f.name == 'z': z_off = f.offset
        if x_off is None:
            return None

        step = msg.point_step
        n = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, step)

        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = np.ndarray(n, dtype=np.float32, buffer=raw[:, x_off:x_off+4].tobytes())
        pts[:, 1] = np.ndarray(n, dtype=np.float32, buffer=raw[:, y_off:y_off+4].tobytes())
        pts[:, 2] = np.ndarray(n, dtype=np.float32, buffer=raw[:, z_off:z_off+4].tobytes())

        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        return pts if len(pts) > 0 else None

    def _det_cb(self, msg):
        """탐지 결과 수신 → 위치 추정 → KF 트래킹 → 발행."""
        stamp = msg.header.stamp
        detections = []

        # depth 이미지 변환
        depth_image = None
        if self.latest_depth is not None:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
            except Exception:
                pass

        # LiDAR 포인트 변환 (numpy 직접 파싱 — pc2.read_points 대비 10x+ 빠름)
        lidar_points = None
        if self.latest_lidar is not None:
            try:
                lidar_points = self._parse_pointcloud(self.latest_lidar)
            except Exception:
                pass

        T = self.transforms

        for det in msg.detections:
            bbox = (
                det.bbox.center.x - det.bbox.size_x * 0.5,
                det.bbox.center.y - det.bbox.size_y * 0.5,
                det.bbox.center.x + det.bbox.size_x * 0.5,
                det.bbox.center.y + det.bbox.size_y * 0.5,
            )
            score = det.results[0].score if det.results else 0.0

            position = None
            distance = None
            m_noise = 1.0  # 기본 측정 노이즈 (높음)

            # 1차: LiDAR 투영 (min_points 이상일 때만 유효)
            source = "none"
            num_pts = 0
            if lidar_points is not None:
                pos_lidar, dist_lidar, num_pts = estimate_position_from_lidar(
                    lidar_points, T["T_lidar_to_color"], T["T_lidar_body"],
                    self.K_color, bbox, self.img_w, self.img_h,
                    self.sample_count, self.outlier_thresh,
                    self.ground_height, self.roof_height,
                    self.min_lidar_points,
                )
                if pos_lidar is not None:
                    position = pos_lidar
                    distance = dist_lidar
                    source = "lidar"
                    m_noise = compute_measurement_noise(
                        num_pts, source="lidar",
                        min_points=self.min_lidar_points,
                        max_points=self.max_lidar_points,
                    )

            # 2차: LiDAR 포인트 부족 시 Depth fallback (높은 측정 노이즈)
            if position is None and depth_image is not None:
                pos_depth, dist_depth = estimate_position_from_depth(
                    depth_image, bbox, self.K_depth, T["T_depth_body"],
                    self.depth_scale, self.depth_min, self.depth_max, self.depth_margin,
                )
                if pos_depth is not None:
                    position = pos_depth
                    distance = dist_depth
                    source = "depth"
                    m_noise = 0.5  # LiDAR(0.1~0.4)보다 높은 노이즈

            rospy.loginfo_throttle(1.0,
                f"[DEBUG] bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) "
                f"lidar_pts={num_pts} src={source} "
                f"pos={position if position is None else f'({position[0]:.2f},{position[1]:.2f},{position[2]:.2f})'} "
                f"dist={distance}")

            if position is not None:
                detections.append((position, score, distance, m_noise))

        # 트래커 업데이트 (Kalman Filter predict + update)
        n_before = len(self.tracker.tracks)
        tracks = self.tracker.update(detections, stamp)
        n_after = len(tracks)
        rospy.loginfo_throttle(1.0,
            f"[DEBUG] dets={len(detections)} tracks_before={n_before} tracks_after={n_after} "
            f"ids=[{','.join(str(t.id) for t in tracks)}]")

        # TrackedPersonArray 메시지 생성
        arr = TrackedPersonArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "base_link"

        for t in tracks:
            p = TrackedPerson()
            p.header.stamp = stamp
            p.header.frame_id = "base_link"
            p.id = t.id
            p.class_id = "person"
            p.score = t.score
            p.position.x = t.position[0]
            p.position.y = t.position[1]
            p.position.z = t.position[2]
            p.velocity.x = t.velocity[0]
            p.velocity.y = t.velocity[1]
            p.velocity.z = t.velocity[2]
            p.distance = np.linalg.norm(t.position[:2])
            p.confidence = t.score
            p.valid = t.valid
            arr.persons.append(p)

        self.pub.publish(arr)
        self._publish_markers(tracks, stamp)

    def _publish_markers(self, tracks, stamp):
        """RViz 시각화용 MarkerArray 발행."""
        if self.marker_pub.get_num_connections() == 0:
            return

        ma = MarkerArray()
        marker_id = 0

        for t in tracks:
            speed = np.linalg.norm(t.velocity[:2])
            dist = np.linalg.norm(t.position[:2])

            # 위치 구체
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "base_link"
            m.ns = "person_sphere"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = t.position[0]
            m.pose.position.y = t.position[1]
            m.pose.position.z = t.position[2]
            m.pose.orientation.w = 1.0
            m.scale.x = 0.4
            m.scale.y = 0.4
            m.scale.z = 0.4
            if dist < 2.0:
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            elif dist < 3.0:
                m.color.r, m.color.g, m.color.b = 1.0, 0.65, 0.0
            else:
                m.color.r, m.color.g, m.color.b = 0.0, 1.0, 0.0
            m.color.a = 1.0 if t.valid else 0.4
            m.lifetime = rospy.Duration(0.5)
            ma.markers.append(m)
            marker_id += 1

            # ID / 거리 / 속도 텍스트
            txt = Marker()
            txt.header.stamp = stamp
            txt.header.frame_id = "base_link"
            txt.ns = "person_text"
            txt.id = marker_id
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = t.position[0]
            txt.pose.position.y = t.position[1]
            txt.pose.position.z = t.position[2] + 0.5
            txt.scale.z = 0.25
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            src = "KF" if t.hit_count > 1 else "new"
            txt.text = f"ID:{t.id} D:{dist:.1f}m V:{speed:.1f}m/s [{src}]"
            txt.lifetime = rospy.Duration(0.5)
            ma.markers.append(txt)
            marker_id += 1

            # 속도 화살표
            if speed > 0.1:
                arrow = Marker()
                arrow.header.stamp = stamp
                arrow.header.frame_id = "base_link"
                arrow.ns = "person_velocity"
                arrow.id = marker_id
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.points = [
                    Point(t.position[0], t.position[1], t.position[2]),
                    Point(t.position[0] + t.velocity[0],
                          t.position[1] + t.velocity[1],
                          t.position[2]),
                ]
                arrow.scale.x = 0.05
                arrow.scale.y = 0.1
                arrow.color.r, arrow.color.g, arrow.color.b = 0.0, 0.8, 1.0
                arrow.color.a = 0.8
                arrow.lifetime = rospy.Duration(0.5)
                ma.markers.append(arrow)
                marker_id += 1

        self.marker_pub.publish(ma)


if __name__ == "__main__":
    try:
        PersonTrackerNode()
    except rospy.ROSInterruptException:
        pass
