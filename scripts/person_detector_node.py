#!/home/irop/projects/vision_robot/venv/bin/python3
# -*- coding: utf-8 -*-

import sys
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
from ultralytics import YOLO
# torchvision C++ ops가 NVIDIA torch와 비호환 → ultralytics 자체 TorchNMS 사용하도록 제거
sys.modules.pop("torchvision", None)


class PersonDetectorNode:
    """RGB 이미지에서 사람(class 0)만 탐지하여 Detection2DArray로 발행하는 노드.
    N 프레임 동안 미탐지된 객체를 유지하는 frame buffer 로직 포함.
    """

    def __init__(self):
        rospy.init_node("person_detector_node", anonymous=False)

        # 파라미터
        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        engine_path = rospy.get_param("~engine_path", "")
        self.conf = rospy.get_param("~confidence_threshold", 0.5)
        self.iou = rospy.get_param("~nms_threshold", 0.45)
        self.imgsz = rospy.get_param("~input_size", 640)
        self.target_class = rospy.get_param("~target_class", 0)  # person
        self.frame_buffer_n = rospy.get_param("~frame_buffer_n", 3)

        if not engine_path:
            rospy.logfatal("engine_path 파라미터가 설정되지 않았습니다.")
            rospy.signal_shutdown("engine_path required")
            return

        # YOLO 모델 로드
        rospy.loginfo("YOLO 모델 로드 중: %s", engine_path)
        self.model = YOLO(engine_path)
        rospy.loginfo("YOLO 모델 로드 완료")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_stamp = None
        self.last_processed_stamp = None

        # N 프레임 유지 버퍼: list of (bbox, score, remaining_frames)
        self.frame_buffer = []

        # Publisher / Subscriber
        self.pub = rospy.Publisher("/detection/persons", Detection2DArray, queue_size=10)
        self.debug_pub = rospy.Publisher("/detection/debug_image", Image, queue_size=1)
        self.sub = rospy.Subscriber(
            image_topic, Image, self._image_cb, queue_size=1, buff_size=2**24
        )

        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self._process()
            rate.sleep()

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def _image_cb(self, msg):
        self.latest_frame = msg
        self.latest_stamp = msg.header.stamp

    # ------------------------------------------------------------------ #
    # Main processing
    # ------------------------------------------------------------------ #
    def _process(self):
        if self.latest_frame is None:
            return
        if self.last_processed_stamp == self.latest_stamp:
            return

        self.last_processed_stamp = self.latest_stamp
        frame_msg = self.latest_frame

        try:
            cv_image = self.bridge.imgmsg_to_cv2(frame_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("이미지 변환 실패: %s", e)
            return

        # YOLO 추론
        results = self.model.predict(
            source=cv_image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        # 현재 프레임 탐지 결과에서 person만 필터링
        current_detections = []  # list of (x1, y1, x2, y2, score)
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != self.target_class:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0])
                    current_detections.append((x1, y1, x2, y2, score))

        # N 프레임 버퍼 업데이트
        merged = self._update_frame_buffer(current_detections)

        # Detection2DArray 메시지 생성 및 발행
        det_array = Detection2DArray()
        det_array.header = frame_msg.header

        for (x1, y1, x2, y2, score) in merged:
            det = Detection2D()
            det.header = frame_msg.header

            bbox = BoundingBox2D()
            bbox.center.x = (x1 + x2) * 0.5
            bbox.center.y = (y1 + y2) * 0.5
            bbox.size_x = max(0.0, x2 - x1)
            bbox.size_y = max(0.0, y2 - y1)
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.id = self.target_class
            hyp.score = score
            det.results.append(hyp)

            det_array.detections.append(det)

        self.pub.publish(det_array)

        # RViz 디버그 이미지 발행
        if self.debug_pub.get_num_connections() > 0:
            debug_img = cv_image.copy()
            for (x1, y1, x2, y2, score) in merged:
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(debug_img, f"person {score:.2f}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # N 프레임 유지 (IoU 기반 frame buffer)
    # ------------------------------------------------------------------ #
    def _update_frame_buffer(self, current_detections):
        """현재 탐지 결과와 버퍼를 매칭하여 병합.
        - 현재 탐지된 객체: 버퍼에 추가/갱신 (remaining = N)
        - 미탐지된 기존 객체: remaining -= 1, 0이면 제거
        """
        matched_buffer_indices = set()
        new_buffer = []

        for (x1, y1, x2, y2, score) in current_detections:
            best_iou = 0.0
            best_idx = -1
            for i, (bx1, by1, bx2, by2, bscore, _remaining) in enumerate(self.frame_buffer):
                iou_val = self._compute_iou(x1, y1, x2, y2, bx1, by1, bx2, by2)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i

            if best_iou > 0.3 and best_idx >= 0:
                matched_buffer_indices.add(best_idx)

            # 현재 탐지는 항상 버퍼에 추가 (최신 정보로 갱신)
            new_buffer.append((x1, y1, x2, y2, score, self.frame_buffer_n))

        # 매칭되지 않은 기존 버퍼 항목 유지 (remaining 감소)
        for i, (bx1, by1, bx2, by2, bscore, remaining) in enumerate(self.frame_buffer):
            if i in matched_buffer_indices:
                continue
            remaining -= 1
            if remaining > 0:
                new_buffer.append((bx1, by1, bx2, by2, bscore, remaining))

        self.frame_buffer = new_buffer

        # 반환: (x1, y1, x2, y2, score) 목록
        return [(x1, y1, x2, y2, sc) for (x1, y1, x2, y2, sc, _r) in self.frame_buffer]

    @staticmethod
    def _compute_iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b):
        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union


if __name__ == "__main__":
    try:
        PersonDetectorNode()
    except rospy.ROSInterruptException:
        pass
