cmake_minimum_required(VERSION 3.0.2)
project(vision_robot)

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  vision_msgs
  cv_bridge
  tf
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  TrackedPerson.msg
  TrackedPersonArray.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  vision_msgs
  cv_bridge
  tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
