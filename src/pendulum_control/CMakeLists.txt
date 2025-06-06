cmake_minimum_required(VERSION 3.5)
project(pendulum_control)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(Eigen3 REQUIRED)


add_executable(pendulum_sim src/pendulum_sim.cpp)
ament_target_dependencies(pendulum_sim rclcpp std_msgs geometry_msgs)

add_executable(pid_controller src/pid_controller.cpp)
ament_target_dependencies(pid_controller rclcpp std_msgs geometry_msgs control_msgs Eigen3)

add_executable(nn_controller src/nn_controller.cpp)
ament_target_dependencies(nn_controller rclcpp std_msgs geometry_msgs control_msgs Eigen3)

add_executable(ac_controller src/actor_critic_controller.cpp)
ament_target_dependencies(ac_controller rclcpp std_msgs geometry_msgs control_msgs Eigen3)
# add_executable(data_logger src/data_logger.cpp)
# ament_target_dependencies(data_logger rclcpp std_msgs geometry_msgs)
include_directories(
  /home/santhosh/headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

target_include_directories(pendulum_sim PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(pid_controller PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(nn_controller PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_include_directories(ac_controller PRIVATE ${EIGEN3_INCLUDE_DIRS})


install(TARGETS
  pendulum_sim
  pid_controller
  nn_controller
  ac_controller
  # data_logger
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
ament_package()
