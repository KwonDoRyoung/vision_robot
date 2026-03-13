#!/home/irop/projects/vision_robot/venv/bin/python3
# -*- coding: utf-8 -*-
"""Phase 4: 사람 회피 정책 노드.

파이프라인: move_base → /cmd_vel_move → [이 노드] → /cmd_vel_safe → safety_node → /cmd_vel

/tracked_persons 에서 사람 위치/속도를 받아,
위험도에 따라 move_base의 cmd_vel을 감속/정지/우회 수정하여 출력.

위험 영역:
    - 정면 ±front_angle 내 접근하는 객체만 위험으로 판단
    - critical (< 1m): 정지
    - warning (< 2m): 크게 감속 + 반대 방향 회전
    - caution (< 3m): 감속
    - 그 외: 원본 cmd_vel 통과

Subscribe:
    /cmd_vel_move (geometry_msgs/Twist)
    /tracked_persons (vision_robot/TrackedPersonArray)

Publish:
    /cmd_vel_safe (geometry_msgs/Twist)
    /avoidance/status (std_msgs/String) - 디버그용
    /avoidance/markers (visualization_msgs/MarkerArray) - RViz 시각화
"""
import math
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from vision_robot.msg import TrackedPersonArray


# 정책 상태
NORMAL = "NORMAL"
CAUTION = "CAUTION"
WARNING = "WARNING"
CRITICAL = "CRITICAL"


class AvoidancePolicyNode:
    def __init__(self):
        rospy.init_node("avoidance_policy_node", anonymous=False)

        # 토픽 파라미터
        input_topic = rospy.get_param("~policy/input_cmd_topic", "/cmd_vel_move")
        output_topic = rospy.get_param("~policy/output_cmd_topic", "/cmd_vel_safe")

        # 정책 파라미터
        self.front_angle = math.radians(rospy.get_param("~policy/front_angle", 30.0))
        self.dist_caution = rospy.get_param("~policy/distance_caution", 3.0)
        self.dist_warning = rospy.get_param("~policy/distance_warning", 2.0)
        self.dist_critical = rospy.get_param("~policy/distance_critical", 1.0)
        self.scale_caution = rospy.get_param("~policy/speed_scale_caution", 0.6)
        self.scale_warning = rospy.get_param("~policy/speed_scale_warning", 0.3)
        self.max_angular = rospy.get_param("~policy/max_angular_avoid", 0.5)
        rate_hz = rospy.get_param("~policy/rate", 20)

        # 상태
        self.latest_cmd = Twist()  # move_base 원본
        self.latest_persons = None
        self.cmd_stamp = rospy.Time.now()

        # Publisher
        self.pub_cmd = rospy.Publisher(output_topic, Twist, queue_size=1)
        self.pub_status = rospy.Publisher("/avoidance/status", String, queue_size=1)
        self.pub_marker = rospy.Publisher("/avoidance/markers", MarkerArray, queue_size=1)

        # Subscriber
        rospy.Subscriber(input_topic, Twist, self._cmd_cb, queue_size=1)
        rospy.Subscriber("/tracked_persons", TrackedPersonArray, self._person_cb, queue_size=1)

        rospy.loginfo(
            f"AvoidancePolicyNode 시작: {input_topic} -> {output_topic} "
            f"(front=±{math.degrees(self.front_angle):.0f}° "
            f"dist={self.dist_critical}/{self.dist_warning}/{self.dist_caution}m)"
        )

        # 타이머 기반 실행 (일정 주기)
        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._timer_cb)
        rospy.spin()

    def _cmd_cb(self, msg):
        self.latest_cmd = msg
        self.cmd_stamp = rospy.Time.now()

    def _person_cb(self, msg):
        self.latest_persons = msg

    def _evaluate_threat(self, person):
        """사람의 위협 수준 평가.

        Returns:
            (level, distance, angle, approaching) - 위험 레벨, 거리, 각도, 접근 여부
        """
        px, py = person.position.x, person.position.y
        vx, vy = person.velocity.x, person.velocity.y

        # 수평 거리 (body 좌표: x=전방, y=좌측)
        dist = math.sqrt(px * px + py * py)
        if dist < 0.01:
            return CRITICAL, dist, 0.0, True

        # 로봇 기준 각도 (0=정면, +좌측, -우측)
        angle = math.atan2(py, px)

        # 접근 속도: 사람→로봇 방향 속도 성분 (음수=접근)
        # 로봇 방향 단위 벡터 = (-px, -py) / dist
        approach_speed = -(vx * px + vy * py) / dist

        # 정면 영역 내 + 접근 중인 경우만 위험 판단
        in_front = abs(angle) < self.front_angle
        approaching = approach_speed > 0.05  # 0.05 m/s 이상으로 접근

        # 정면이 아니거나 멀어지는 경우: 거리만으로 판단 (critical만)
        if not in_front:
            if dist < self.dist_critical:
                return WARNING, dist, angle, False
            return NORMAL, dist, angle, False

        if not approaching:
            # 정면이지만 접근하지 않음 → 거리 기반 경계만
            if dist < self.dist_critical:
                return CRITICAL, dist, angle, False
            elif dist < self.dist_warning:
                return CAUTION, dist, angle, False
            return NORMAL, dist, angle, False

        # 정면 + 접근 중 → 풀 위험 평가
        if dist < self.dist_critical:
            return CRITICAL, dist, angle, True
        elif dist < self.dist_warning:
            return WARNING, dist, angle, True
        elif dist < self.dist_caution:
            return CAUTION, dist, angle, True

        return NORMAL, dist, angle, True

    def _timer_cb(self, event):
        """주기적 정책 실행."""
        cmd_out = Twist()
        cmd_in = self.latest_cmd
        policy_state = NORMAL
        threat_info = ""

        # cmd_vel_move 가 오래되면 (1초 이상) 정지
        if (rospy.Time.now() - self.cmd_stamp).to_sec() > 1.0:
            cmd_in = Twist()

        # 사람 정보 평가
        worst_level = NORMAL
        worst_dist = float('inf')
        worst_angle = 0.0
        worst_approaching = False
        level_priority = {NORMAL: 0, CAUTION: 1, WARNING: 2, CRITICAL: 3}

        if self.latest_persons is not None:
            for person in self.latest_persons.persons:
                if not person.valid:
                    continue
                level, dist, angle, approaching = self._evaluate_threat(person)
                if level_priority[level] > level_priority[worst_level]:
                    worst_level = level
                    worst_dist = dist
                    worst_angle = angle
                    worst_approaching = approaching

        policy_state = worst_level

        if policy_state == CRITICAL:
            # 정지
            cmd_out.linear.x = 0.0
            cmd_out.linear.y = 0.0
            cmd_out.angular.z = 0.0
            threat_info = f"STOP dist={worst_dist:.2f}m"

        elif policy_state == WARNING:
            # 크게 감속 + 반대 방향 회전
            cmd_out.linear.x = cmd_in.linear.x * self.scale_warning
            cmd_out.linear.y = cmd_in.linear.y * self.scale_warning

            # 사람이 좌측(angle>0)이면 우회전(-), 우측(angle<0)이면 좌회전(+)
            avoid_dir = -1.0 if worst_angle > 0 else 1.0
            cmd_out.angular.z = cmd_in.angular.z + avoid_dir * self.max_angular
            # 회전 속도 제한
            cmd_out.angular.z = max(-self.max_angular, min(self.max_angular, cmd_out.angular.z))
            threat_info = f"AVOID dist={worst_dist:.2f}m ang={math.degrees(worst_angle):.0f}°"

        elif policy_state == CAUTION:
            # 감속
            cmd_out.linear.x = cmd_in.linear.x * self.scale_caution
            cmd_out.linear.y = cmd_in.linear.y * self.scale_caution
            cmd_out.angular.z = cmd_in.angular.z
            threat_info = f"SLOW dist={worst_dist:.2f}m"

        else:
            # 정상 통과
            cmd_out = cmd_in

        self.pub_cmd.publish(cmd_out)

        # 상태 발행
        status_msg = String()
        status_msg.data = f"{policy_state} {threat_info}"
        self.pub_status.publish(status_msg)

        rospy.loginfo_throttle(1.0,
            f"[AVOIDANCE] {policy_state} | "
            f"in=({cmd_in.linear.x:.2f}, {cmd_in.angular.z:.2f}) "
            f"out=({cmd_out.linear.x:.2f}, {cmd_out.angular.z:.2f}) "
            f"{threat_info}")

        # RViz 시각화
        self._publish_markers(policy_state, worst_dist, worst_angle)

    def _publish_markers(self, state, dist, angle):
        """위험 영역 시각화."""
        if self.pub_marker.get_num_connections() == 0:
            return

        ma = MarkerArray()

        # 위험 영역 부채꼴 (정면 ±front_angle, 3단계 거리)
        zones = [
            (self.dist_caution, 0.1, 1.0, 1.0, 0.0, 0.15),   # 노란색 (주의)
            (self.dist_warning, 0.1, 1.0, 0.5, 0.0, 0.2),    # 주황색 (경고)
            (self.dist_critical, 0.1, 1.0, 0.0, 0.0, 0.25),  # 빨간색 (긴급)
        ]

        for i, (radius, height, r, g, b, a) in enumerate(zones):
            m = Marker()
            m.header.stamp = rospy.Time.now()
            m.header.frame_id = "base_link"
            m.ns = "avoidance_zone"
            m.id = i
            m.type = Marker.TRIANGLE_LIST
            m.action = Marker.ADD

            # 부채꼴 삼각형 (정면 ±front_angle)
            n_segments = 12
            angle_start = -self.front_angle
            angle_step = 2.0 * self.front_angle / n_segments

            for j in range(n_segments):
                a1 = angle_start + j * angle_step
                a2 = angle_start + (j + 1) * angle_step

                # 원점
                p0 = Point(0, 0, height)
                # 호 위의 두 점
                p1 = Point(radius * math.cos(a1), radius * math.sin(a1), height)
                p2 = Point(radius * math.cos(a2), radius * math.sin(a2), height)

                m.points.extend([p0, p1, p2])

            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a

            # 현재 활성 영역은 더 밝게
            if (state == CAUTION and i == 0) or \
               (state == WARNING and i == 1) or \
               (state == CRITICAL and i == 2):
                m.color.a = 0.6

            m.lifetime = rospy.Duration(0.2)
            ma.markers.append(m)

        # 현재 상태 텍스트
        txt = Marker()
        txt.header.stamp = rospy.Time.now()
        txt.header.frame_id = "base_link"
        txt.ns = "avoidance_text"
        txt.id = 100
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = 0.0
        txt.pose.position.y = 0.0
        txt.pose.position.z = 1.5
        txt.scale.z = 0.3
        if state == CRITICAL:
            txt.color.r, txt.color.g, txt.color.b = 1.0, 0.0, 0.0
        elif state == WARNING:
            txt.color.r, txt.color.g, txt.color.b = 1.0, 0.5, 0.0
        elif state == CAUTION:
            txt.color.r, txt.color.g, txt.color.b = 1.0, 1.0, 0.0
        else:
            txt.color.r, txt.color.g, txt.color.b = 0.0, 1.0, 0.0
        txt.color.a = 1.0
        txt.text = state
        txt.lifetime = rospy.Duration(0.2)
        ma.markers.append(txt)

        self.pub_marker.publish(ma)


if __name__ == "__main__":
    try:
        AvoidancePolicyNode()
    except rospy.ROSInterruptException:
        pass
