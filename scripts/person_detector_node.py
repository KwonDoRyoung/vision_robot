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
    """RGB 이미지에서 사람(class 0)을 탐지+추적(ByteTrack)하여 Detection2DArray로 발행.

    hyp.id = ByteTrack track_id (사람 class 고정이므로 class_id 대신 track_id 전달)
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

        if not engine_path:
            rospy.logfatal("engine_path 파라미터가 설정되지 않았습니다.")
            rospy.signal_shutdown("engine_path required")
            return

        # YOLO 모델 로드
        rospy.loginfo("YOLO 모델 로드 중: %s", engine_path)
        self.model = YOLO(engine_path)
        rospy.loginfo("YOLO 모델 로드 완료 (ByteTrack 추적 활성화)")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_stamp = None
        self.last_processed_stamp = None

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

    def _image_cb(self, msg):
        self.latest_frame = msg
        self.latest_stamp = msg.header.stamp

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

        # YOLO 추론 + ByteTrack 추적
        results = self.model.track(
            source=cv_image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # 결과 파싱: person만 필터링, track_id 포함
        tracked_dets = []  # (x1, y1, x2, y2, score, track_id)
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                has_ids = boxes.id is not None
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    if cls_id != self.target_class:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0])
                    track_id = int(boxes.id[i]) if has_ids else -1
                    tracked_dets.append((x1, y1, x2, y2, score, track_id))

        # Detection2DArray 메시지 생성 및 발행
        det_array = Detection2DArray()
        det_array.header = frame_msg.header

        for (x1, y1, x2, y2, score, track_id) in tracked_dets:
            det = Detection2D()
            det.header = frame_msg.header

            bbox = BoundingBox2D()
            bbox.center.x = (x1 + x2) * 0.5
            bbox.center.y = (y1 + y2) * 0.5
            bbox.size_x = max(0.0, x2 - x1)
            bbox.size_y = max(0.0, y2 - y1)
            det.bbox = bbox

            hyp = ObjectHypothesisWithPose()
            hyp.id = track_id  # ByteTrack ID (person class 고정)
            hyp.score = score
            det.results.append(hyp)

            det_array.detections.append(det)

        self.pub.publish(det_array)

        # RViz 디버그 이미지 발행
        if self.debug_pub.get_num_connections() > 0:
            debug_img = cv_image.copy()
            for (x1, y1, x2, y2, score, track_id) in tracked_dets:
                color = (0, 255, 0)
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{track_id} {score:.2f}" if track_id >= 0 else f"person {score:.2f}"
                cv2.putText(debug_img, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            except Exception:
                pass


if __name__ == "__main__":
    try:
        PersonDetectorNode()
    except rospy.ROSInterruptException:
        pass
