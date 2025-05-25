#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64.hpp>
#include <control_msgs/msg/state_action_pair.hpp>
#include <Eigen/Dense>
class PIDController : public rclcpp::Node
{
public:
    PIDController() : Node("pid_controller")
    {
        state_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "pendulum/state", 10,
            std::bind(&PIDController::state_callback, this, std::placeholders::_1));

        control_pub_ = this->create_publisher<std_msgs::msg::Float64>("pendulum/control", 10);
        action_label_pub_ = this->create_publisher<control_msgs::msg::StateActionPair>("pendulum/action_label", 10);

        kp_ = 20.0;
        kd_ = 5.0;
        sap_msg.state.resize(2); sap_msg.action.resize(1);
    }

private:
    double theta = 0.0;
    double theta_dot = 0.0;

    double control = 0.0;

    std_msgs::msg::Float64 control_msg;
    control_msgs::msg::StateActionPair sap_msg;
    
    void state_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        theta = msg->x;
        theta_dot = msg->y;

        control = -kp_ * theta - kd_ * theta_dot;

        control_msg.data = control;

        control_pub_->publish(control_msg);
        sap_msg.state[0] = theta; sap_msg.state[1] = theta_dot; sap_msg.action[0] = control;
        action_label_pub_->publish(sap_msg);
    }

    double kp_, kd_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr state_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr control_pub_;
    rclcpp::Publisher<control_msgs::msg::StateActionPair>::SharedPtr action_label_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PIDController>());
    rclcpp::shutdown();
    return 0;
}
