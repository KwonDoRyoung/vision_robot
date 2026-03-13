#!/home/irop/projects/vision_robot/venv/bin/python3
# -*- coding: utf-8 -*-
"""Phase 3: ByteTrack ID 기반 LiDAR/Depth 융합 사람 위치·속도 추정 노드.

ByteTrack이 2D 이미지에서 ID 추적 → 이 노드는 ID별 3D 위치 추정 + EMA 스무딩.
거리 기반 서브 박스로 배경 LiDAR 포인트 배제.

Subscribe:
    /detection/persons (vision_msgs/Detection2DArray) - hyp.id = ByteTrack track_id
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

from vision_robot.msg import TrackedPerson, TrackedPersonArray
from vision_robot.config import get_transforms, get_color_intrinsics, get_depth_intrinsics
from vision_robot.projection import (
    compute_sub_bbox,
    estimate_position_from_depth,
    estimate_position_from_lidar,
)
from vision_robot.tracker import TrackedPersonStore


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
        self.min_lidar_points = rospy.get_param("~fusion/min_lidar_points", 35)

        # 서브 박스 파라미터
        self.sub_close_ratio = rospy.get_param("~fusion/sub_close_ratio", 0.35)
        self.sub_far_ratio = rospy.get_param("~fusion/sub_far_ratio", 0.55)
        self.sub_distance_threshold = rospy.get_param("~fusion/sub_distance_threshold", 2.0)

        # KF 스무딩 파라미터
        max_lost = rospy.get_param("~tracker/max_lost_frames", 5)
        q_pos = rospy.get_param("~tracker/process_noise_pos", 0.05)
        q_vel = rospy.get_param("~tracker/process_noise_vel", 0.5)
        r_pos = rospy.get_param("~tracker/measurement_noise", 0.3)

        # 변환 행렬 / intrinsics
        self.transforms = get_transforms()
        self.K_color = get_color_intrinsics()
        self.K_depth = get_depth_intrinsics()

        # ByteTrack ID 기반 KF 스무딩 저장소
        self.store = TrackedPersonStore(max_lost, q_pos, q_vel, r_pos)

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

        rospy.loginfo("PersonTrackerNode 시작 (ByteTrack ID + EMA 스무딩)")
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
        """탐지 결과 수신 → 서브 박스 → 3D 위치 추정 → EMA 스무딩 → 발행."""
        stamp = msg.header.stamp

        # depth 이미지 변환
        depth_image = None
        if self.latest_depth is not None:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding="passthrough")
            except Exception:
                pass

        # LiDAR 포인트 변환
        lidar_points = None
        if self.latest_lidar is not None:
            try:
                lidar_points = self._parse_pointcloud(self.latest_lidar)
            except Exception:
                pass

        T = self.transforms
        active_ids = set()

        for det in msg.detections:
            track_id = int(det.results[0].id) if det.results else -1
            score = det.results[0].score if det.results else 0.0

            # track_id 없으면 무시 (ByteTrack 미할당)
            if track_id < 0:
                continue

            bbox = (
                det.bbox.center.x - det.bbox.size_x * 0.5,
                det.bbox.center.y - det.bbox.size_y * 0.5,
                det.bbox.center.x + det.bbox.size_x * 0.5,
                det.bbox.center.y + det.bbox.size_y * 0.5,
            )

            # 이전 거리로 서브 박스 계산
            prev_dist = self.store.get_prev_distance(track_id)
            sub_bbox = compute_sub_bbox(
                bbox, prev_dist,
                self.sub_close_ratio, self.sub_far_ratio,
                self.sub_distance_threshold,
            )

            position = None
            distance = None
            source = "none"
            num_pts = 0

            # 1차: LiDAR 투영 (서브 박스 사용)
            if lidar_points is not None:
                pos_lidar, dist_lidar, num_pts = estimate_position_from_lidar(
                    lidar_points, T["T_lidar_to_color"], T["T_lidar_body"],
                    self.K_color, sub_bbox, self.img_w, self.img_h,
                    self.sample_count, self.outlier_thresh,
                    self.ground_height, self.roof_height,
                    self.min_lidar_points,
                )
                if pos_lidar is not None:
                    position = pos_lidar
                    distance = dist_lidar
                    source = "lidar"

            # 2차: LiDAR 부족 시 Depth fallback (원본 bbox 사용)
            if position is None and depth_image is not None:
                pos_depth, dist_depth = estimate_position_from_depth(
                    depth_image, bbox, self.K_depth, T["T_depth_body"],
                    self.depth_scale, self.depth_min, self.depth_max, self.depth_margin,
                )
                if pos_depth is not None:
                    position = pos_depth
                    distance = dist_depth
                    source = "depth"

            rospy.loginfo_throttle(1.0,
                f"[TRACK] ID:{track_id} sub_bbox=({sub_bbox[0]:.0f},{sub_bbox[1]:.0f},"
                f"{sub_bbox[2]:.0f},{sub_bbox[3]:.0f}) "
                f"lidar_pts={num_pts} src={source} "
                f"pos={position if position is None else f'({position[0]:.2f},{position[1]:.2f},{position[2]:.2f})'} "
                f"dist={'None' if distance is None else f'{distance:.2f}'}")

            if position is not None:
                self.store.update(track_id, position, stamp, score)
                active_ids.add(track_id)

        # lost 트랙 처리 및 정리
        self.store.mark_lost_and_prune(active_ids)

        # TrackedPersonArray 메시지 생성
        arr = TrackedPersonArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "base_link"

        all_tracks = self.store.get_all_tracks()
        for tid, s in all_tracks:
            p = TrackedPerson()
            p.header.stamp = stamp
            p.header.frame_id = "base_link"
            p.id = tid
            p.class_id = "person"
            p.score = s.score
            p.position.x = s.position[0]
            p.position.y = s.position[1]
            p.position.z = s.position[2]
            p.velocity.x = s.velocity[0]
            p.velocity.y = s.velocity[1]
            p.velocity.z = s.velocity[2]
            p.distance = s.distance
            p.confidence = s.score
            p.valid = s.valid
            arr.persons.append(p)

        self.pub.publish(arr)
        self._publish_markers(all_tracks, stamp)

    def _publish_markers(self, tracks, stamp):
        """RViz 시각화용 MarkerArray 발행."""
        if self.marker_pub.get_num_connections() == 0:
            return

        ma = MarkerArray()
        marker_id = 0

        for tid, s in tracks:
            if not s.valid:
                continue

            speed = float(np.linalg.norm(s.velocity[:2]))
            dist = s.distance

            # 위치 구체
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "base_link"
            m.ns = "person_sphere"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = s.position[0]
            m.pose.position.y = s.position[1]
            m.pose.position.z = s.position[2]
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
            m.color.a = 1.0
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
            txt.pose.position.x = s.position[0]
            txt.pose.position.y = s.position[1]
            txt.pose.position.z = s.position[2] + 0.5
            txt.scale.z = 0.25
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f"ID:{tid} D:{dist:.1f}m V:{speed:.1f}m/s"
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
                    Point(s.position[0], s.position[1], s.position[2]),
                    Point(s.position[0] + s.velocity[0],
                          s.position[1] + s.velocity[1],
                          s.position[2]),
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
