# Vision Robot - 개발 계획서

## 프로젝트 개요
카메라(RGB/Depth)와 LiDAR를 이용한 실시간 자율주행 로봇 구현
동적 객체(사람)를 감지하고 위치/속도를 추정하여, 정책 기반 회피 주행을 수행한다.

---

## 시스템 구성

| 구분 | 사양 |
|------|------|
| OS | Ubuntu 20.04 |
| 미들웨어 | ROS1 Noetic |
| 컴퓨팅 | Jetson AGX Orin 32GB |
| LiDAR | Livox Mid-360 x 2 (정합 완료) |
| 카메라 | Intel RealSense D455 |

### 좌표계 (로봇베이스 중심 기준)
- 전면 카메라 RGB: (0.28, 0.01, 0.115)
- 전면 카메라 Depth: (0.28, -0.01, 0.115)
- LiDAR 정합 중심: (0.0, 0.0, 0.22)

### ROS 토픽
- RGB: `/camera/color/image_raw`
- Depth: `/camera/depth/image_rect_raw`
- LiDAR: `/rslidar_points`

---

## 패키지 구조

본 프로젝트는 **catkin 패키지**로 구성하며, 심링크를 통해 catkin 워크스페이스에 연결한다.

```
프로젝트 경로: /home/irop/projects/vision_robot/
심링크: /home/irop/catkin_ws/src/vision_robot → /home/irop/projects/vision_robot/
```

### 심링크 생성 및 빌드
```bash
ln -s /home/irop/projects/vision_robot /home/irop/catkin_ws/src/vision_robot
cd /home/irop/catkin_ws && catkin_make
source devel/setup.bash
```

### 디렉토리 구조
```
vision_robot/
├── CMakeLists.txt                 # catkin 빌드 설정
├── package.xml                    # catkin 패키지 매니페스트
├── .gitignore
├── RESEARCH.md
├── RESEARCH_NEW.md
├── msg/
│   ├── TrackedPerson.msg          # 개별 추적 객체 메시지
│   └── TrackedPersonArray.msg     # 추적 객체 배열 메시지
├── launch/
│   └── vision_robot.launch        # 전체 노드 실행
├── scripts/                       # 실행 가능한 ROS 노드 (#!/usr/bin/env python3)
│   ├── person_detector_node.py    # Phase 2: YOLO 객체 탐지
│   ├── person_tracker_node.py     # Phase 3: 위치/속도 추정
│   └── avoidance_policy_node.py   # Phase 4: 회피 정책
├── src/                           # Python 모듈 (import용, 직접 실행 X)
│   └── vision_robot/
│       ├── __init__.py
│       ├── projection.py          # LiDAR→이미지 투영
│       ├── tracker.py             # 객체 트래킹
│       └── config.py              # 설정값 관리
└── config/
    └── params.yaml                # ROS 파라미터
```

### 커스텀 메시지 정의

**TrackedPerson.msg**
```
std_msgs/Header header
uint32 id
string class_id
float32 score
geometry_msgs/Point position       # 로봇 기준 3D 위치 (x, y, z)
geometry_msgs/Vector3 velocity     # 속도 벡터 (vx, vy, vz)
float64 distance                   # 로봇까지 거리
float64 confidence                 # 탐지 신뢰도
bool valid
```

**TrackedPersonArray.msg**
```
std_msgs/Header header
TrackedPerson[] persons
```

### 기존 패키지와의 관계
- `obj_position_estimation` (RepresentativeObject): 기존 위치 추정 패키지 참고
- `obj_velocity_estimation` (RepresentativeVelocity): 기존 속도 추정 패키지 참고
- 본 패키지는 위 두 패키지의 기능을 **통합**하여 단일 파이프라인으로 구현

---

## 개발 단계

### Phase 1: 환경 구축 및 센서 검증
> 목표: catkin 패키지 구성, 개발 환경 세팅, 센서 데이터 정상 수신 확인

- [ ] catkin 패키지 기본 파일 생성 (`package.xml`, `CMakeLists.txt`)
- [ ] catkin_ws/src에 심링크 생성 및 빌드 확인
- [ ] 커스텀 메시지 정의 및 빌드 확인 (`TrackedPerson.msg`, `TrackedPersonArray.msg`)
- [ ] Python venv 환경 구성 (ROS1 호환)
  - venv 생성 시 `--system-site-packages` 옵션으로 ROS 패키지 접근 유지
- [ ] 필수 패키지 설치 (ultralytics, tensorrt, numpy, opencv)
- [ ] 각 센서 토픽 수신 확인 (rostopic echo)
  - RGB 이미지 정상 수신
  - Depth 이미지 정상 수신
  - LiDAR 포인트클라우드 정상 수신
- [ ] 카메라-LiDAR 간 TF(좌표 변환) 확인
  - 카메라 intrinsic 파라미터 확인 (`/camera/color/camera_info`)
  - LiDAR → 카메라 extrinsic 변환 행렬 산출/검증
- [ ] 센서 데이터 시간 동기화 방식 결정 (approximate time synchronizer)

**검증**:
1. `catkin_make` 빌드 성공
2. `rosmsg show vision_robot/TrackedPerson` 으로 메시지 확인
3. rviz에서 LiDAR 포인트를 카메라 이미지 위에 투영하여 정합 확인

---

### Phase 2: 객체 탐지 노드 개발
> 목표: RGB 이미지에서 사람을 실시간으로 탐지하고 bounding box를 발행

- [ ] YOLOv11n (또는 최신 Ultralytics 모델) 선정 및 TensorRT 변환
  - `.pt` → `.engine` 변환 (Jetson 최적화, FP16)
  - 입력 해상도 결정 (640x640 권장)
  - 참고: 기존 `yolo26n.engine` 파일이 catkin_ws/src에 존재
- [ ] ROS 노드 구현: `scripts/person_detector_node.py`
  - Subscribe: `/camera/color/image_raw` (sensor_msgs/Image)
  - Publish: `/detection/persons` (vision_msgs/Detection2DArray)
  - 클래스 필터링: person(class 0)만 발행
- [ ] N 프레임 이전 결과 유지 로직 구현
  - 현재 프레임에서 미탐지된 객체를 N 프레임(설정 가능) 동안 유지
  - 간단한 IoU 기반 트래킹 또는 frame buffer 방식
- [ ] 성능 측정
  - 추론 속도 목표: 15 FPS 이상
  - GPU 메모리 사용량 확인

**검증**: 사람이 카메라 앞을 지나갈 때 bounding box가 안정적으로 출력되는지 확인

---

### Phase 3: LiDAR-카메라 융합 및 위치/속도 추정
> 목표: 탐지된 사람의 3D 위치와 이동 속도를 추정

#### 3-1. LiDAR 포인트 → 2D 이미지 투영
- [ ] `src/vision_robot/projection.py` 모듈 구현
  - LiDAR 3D 포인트를 카메라 이미지 평면에 투영
  - 변환 순서: LiDAR 좌표 → 로봇 베이스 → 카메라 좌표 → 이미지 좌표
  - T_lidar_to_base, T_base_to_cam, K(intrinsic) 활용
- [ ] bounding box 내부 LiDAR 포인트 필터링
  - 투영된 2D 좌표가 bbox 내부에 있는 포인트만 추출
  - S개 샘플링 (설정 가능, 초기값 30~50개 권장)
  - 이상치 제거: median 기반 필터링 (너무 먼/가까운 포인트 제거)

#### 3-2. 객체 위치 추정
- [ ] 필터링된 포인트의 중앙값(median)으로 객체 3D 위치 산출
  - 로봇 베이스 좌표계 기준 (x, y, z)
- [ ] Depth 카메라 보조 활용 (선택사항)
  - LiDAR 포인트가 부족할 경우 depth로 보완
  - bbox 중심 영역의 depth 값 참조

#### 3-3. 객체 속도 추정
- [ ] `src/vision_robot/tracker.py` 모듈 구현
  - 다중 프레임 간 객체 매칭 (간단한 트래커)
  - 헝가리안 알고리즘 또는 거리 기반 최근접 매칭
  - 객체 ID 부여 및 유지
- [ ] 속도 계산
  - 위치 변화량 / 시간 변화량 = 속도 벡터 (vx, vy)
  - 이동 평균 필터로 노이즈 제거

#### 3-4. ROS 노드 구현: `scripts/person_tracker_node.py`
- [ ] Subscribe: `/detection/persons`, `/rslidar_points`
- [ ] Publish: `/tracked_persons` (vision_robot/TrackedPersonArray)

**검증**: 사람이 걸어갈 때 위치와 속도가 합리적인 값으로 출력되는지 확인 (사람 보행 속도 ~1.2m/s)

---

### Phase 4: 회피 정책(Policy) 노드 개발
> 목표: 추적된 객체 정보를 기반으로 주행 명령을 결정

- [ ] ROS 노드 구현: `scripts/avoidance_policy_node.py`
  - Subscribe: `/tracked_persons` (vision_robot/TrackedPersonArray)
  - Publish: `/policy/cmd` (geometry_msgs/Twist)

- [ ] 위험 영역 정의
  - 정면 영역: 로봇 진행 방향 기준 좌우 ±30° 이내
  - 위험 거리: 단계별 (예: 3m 주의, 2m 경고, 1m 긴급)

- [ ] 정책 룰 구현

  | 상황 | 행동 |
  |------|------|
  | 정면 접근 객체 (거리 < 임계값) | 정지 → 측면 이동 가능 시 이동 |
  | 정면이 아닌 접근 객체 | 객체 반대 방향으로 자연스러운 회전 |
  | 측면 객체 (접근 중) | 거리 유지하며 주행 계속 |
  | 위험 객체 없음 | 정상 주행 |

- [ ] 정책 출력 타입 정의
  - NORMAL: 정상 주행
  - STOP: 정지
  - MOVE_LEFT: 좌측 이동
  - MOVE_RIGHT: 우측 이동
  - ROTATE_LEFT: 좌회전
  - ROTATE_RIGHT: 우회전
  - ROTATE_IN_PLACE: 제자리 회전

**검증**: 시뮬레이션된 객체 데이터로 정책 출력이 올바른지 단위 테스트

---

### Phase 5: 통합 테스트 및 최적화
> 목표: 전체 파이프라인을 연결하고 실시간 동작을 검증

- [ ] 전체 노드 launch 파일 작성 (`launch/vision_robot.launch`)
  - `roslaunch vision_robot vision_robot.launch` 로 실행 가능
- [ ] 파이프라인 지연 시간 측정
  - 목표: 센서 입력 → 정책 출력 200ms 이내
- [ ] 병목 구간 최적화
  - LiDAR 투영 연산 최적화 (numpy vectorize)
  - 불필요한 메시지 복사 제거
- [ ] 실제 환경 테스트
  - 정지 상태에서 사람 접근 시 반응 확인
  - 저속 주행 중 사람 접근 시 반응 확인
- [ ] 파라미터 튜닝
  - N 프레임 유지 수, S 샘플링 수, 위험 거리 임계값 등
- [ ] rosbag 녹화 환경 구성 (추후 진행)

---

## 주의 사항
1. **catkin 패키지 규칙 준수** - `package.xml`, `CMakeLists.txt` 필수, 기존 패키지 컨벤션 따름
2. **venv 사용** - `python3 -m venv --system-site-packages venv` 로 ROS 패키지 접근 유지
3. **단위/기능별 검증** - 각 Phase 완료 시 반드시 검증 후 다음 단계 진행
4. **수정 범위 제한** - `/home/irop/catkin_ws/src` 의 기존 패키지는 절대 수정 금지, 심링크된 본 프로젝트 내에서만 수정
5. **LiDAR 토픽**: `/rslidar_points`
6. **실행 스크립트** - `scripts/` 디렉토리에 위치, `chmod +x` 필수, shebang `#!/usr/bin/env python3`
7. **모듈 코드** - `src/vision_robot/` 디렉토리에 위치, 노드에서 import하여 사용

---

## 기술적 고려 사항

### LiDAR 3D → 카메라 2D 투영 방식
YOLO bbox는 2D 이미지 평면 위의 사각형이고, LiDAR 포인트는 3D 공간의 점이다.
투영 과정:
1. LiDAR 포인트 P_lidar = (x, y, z)
2. 카메라 좌표로 변환: P_cam = T_lidar_to_cam * P_lidar
3. 이미지 좌표로 투영: p_img = K * P_cam (정규화 후)
4. p_img가 bbox 내부이면 해당 포인트는 그 객체에 속함

이 방식은 **bbox가 2D이므로 깊이 방향으로 여러 객체가 겹칠 수 있다는 한계**가 있다.
→ depth 클러스터링 또는 median 필터로 완화

### 포인트 샘플링 전략
- bbox 내 전체 포인트에서 S개를 랜덤 샘플링하되, depth 기준 이상치 먼저 제거
- 권장 초기값: S = 30~50 (속도와 정확도 균형)
- 실험을 통해 최적값 튜닝

### 실시간성 확보
- YOLO TensorRT FP16: ~15-30 FPS (Jetson AGX Orin)
- LiDAR 투영: numpy 벡터 연산으로 1ms 이내 목표
- 전체 파이프라인: 200ms 이내 목표
