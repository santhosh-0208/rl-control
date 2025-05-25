#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64.hpp>
#include <Eigen/Dense>
#include <control_msgs/msg/state_action_pair.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class simpleNN
{
public:
    simpleNN(int input_size, int hidden_size, int output_size)
    {
        W1 = MatrixXd::Random(input_size, hidden_size);
        b1 = VectorXd::Zero(hidden_size);

        W2 = MatrixXd::Random(hidden_size, output_size);
        b2 = VectorXd::Zero(output_size);

        learning_rate = 0.001;
    }

    double forward(const VectorXd &input)
    {
        z1 = (input.transpose() * W1).transpose() + b1;
        a1 = z1.unaryExpr([](double x)
                          { return 1.0 / (1.0 + std::exp(-x)); });

        double output = (a1.transpose() * W2).sum() + b2(0);
        return output;
    }

    void train(const VectorXd &input, const double &pid_output)
    {
        double nn_output = forward(input);
        double loss = (nn_output - pid_output);

        double d_output = 2.0 * loss;
        VectorXd dW2 = a1 * d_output;

        double db2 = d_output;

        VectorXd da1 = W2 * d_output;
        VectorXd dz1 = da1.array() * (a1.array() * (1.0 - a1.array()));

        MatrixXd dW1 = input * dz1.transpose();
        VectorXd db1 = dz1;

        // Update
        W2 -= learning_rate * dW2;
        b2(0) -= learning_rate * db2;
        W1 -= learning_rate * dW1;
        b1 -= learning_rate * db1;
    }

private:
    MatrixXd W1;
    MatrixXd W2;
    VectorXd b1;
    VectorXd b2;

    MatrixXd z1;
    MatrixXd a1;

    double learning_rate;
};
class NNcontroller : public rclcpp::Node
{
public:
    NNcontroller() : Node("nn_controller"), nn_(2,8,1)
    {
        state_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "pendulum/state", 10,
            std::bind(&NNcontroller::state_callback, this, std::placeholders::_1));

        action_label_sub_ = this->create_subscription<control_msgs::msg::StateActionPair>(
            "pendulum/action_label", 10,
            std::bind(&NNcontroller::action_label_callback, this, std::placeholders::_1));

        nn_control_pub_ = this->create_publisher<std_msgs::msg::Float64>("pendulum/nn_control", 10);
        input.resize(2);
        training_input.resize(2);

    }

private:
    double theta = 0.0;
    double theta_dot = 0.0;

    double nn_control = 0.0;
    VectorXd input;
    double pid_output; 
    VectorXd training_input;

    std_msgs::msg::Float64 control_msg;
    void state_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        theta = msg->x;
        theta_dot = msg->y;
        input << theta, theta_dot;

        nn_control = nn_.forward(input);

        control_msg.data = nn_control;

        nn_control_pub_->publish(control_msg);
    }
    void action_label_callback(const control_msgs::msg::StateActionPair::SharedPtr msg)
    {
        pid_output = msg->action[0];
        training_input << msg->state[0], msg->state[1];
        nn_.train(training_input, pid_output);
    }
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr state_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nn_control_pub_;
    rclcpp::Subscription<control_msgs::msg::StateActionPair>::SharedPtr action_label_sub_;
    simpleNN nn_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NNcontroller>());
    rclcpp::shutdown();
    return 0;
}
