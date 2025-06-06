#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64.hpp>

class PendulumSim : public rclcpp::Node
{
public:
    PendulumSim() : Node("pendulum_sim"), theta_(0.5), theta_dot_(0.0), torque_(0.0)
    {
        state_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>("pendulum/state", 10);
        control_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "pendulum/control", 10,
            std::bind(&PendulumSim::control_callback, this, std::placeholders::_1));

        nn_control_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "pendulum/ac_nn_control", 10,
            std::bind(&PendulumSim::nn_control_callback, this, std::placeholders::_1));
        control_source_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "pendulum/control_source", 10,
            std::bind(&PendulumSim::control_source_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&PendulumSim::update, this));
    }

private:
    void control_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        if (!nn_control)
        {

            torque_ = msg->data;
        }
    }
    void nn_control_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        if (nn_control)
        {

            torque_ = msg->data;
        }
    }
    void control_source_callback(const std_msgs::msg::Float64::SharedPtr msg)
    {
        if (msg->data == 0)
        {
            nn_control = false;
        }
        else
        {
            nn_control = true;
        }
    }
    const double dt = 0.01; // 100Hz
    const double g = 9.81;
    const double l = 1.0;
    double gravity_torque = 0.0;
    double theta_, theta_dot_, torque_;
    int counter = 0;
    int episode_length_counter = 0;
    bool nn_control = true;
    bool episode_failed = false;
    void update()
    {

        gravity_torque = -g * l * std::sin(theta_);
        theta_dot_ += (gravity_torque + torque_) * dt;
        theta_ += theta_dot_ * dt;

        auto msg = geometry_msgs::msg::Vector3();
        msg.x = theta_;
        msg.y = theta_dot_;
        if (fabs(theta_) > 4.0)
        {
            episode_failed = true;
        }
        if (fabs(theta_) < 0.001 && fabs(theta_dot_) < 0.001)
        {
            counter += 1;
        }
        else
        {
            counter = 0;
        }
        msg.z = 0.0;

        episode_length_counter += 1;
        if (counter > 50 || episode_length_counter > 600 || episode_failed)
        {

            theta_ = ((double)rand() / (RAND_MAX)) * 2.0 * M_PI - M_PI;

            theta_dot_ = ((double)rand() / (RAND_MAX)) * 2.0 - 1.0;

            if (episode_failed)
            {
                msg.z = -1.0;
            }
            else
            {
                msg.z = 1.0;
            }
            episode_length_counter = 0;
            counter = 0;
            episode_failed = false;
        }
        state_pub_->publish(msg);
    }

    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr state_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr control_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr nn_control_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr control_source_sub_;

    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PendulumSim>());
    rclcpp::shutdown();
    return 0;
}
