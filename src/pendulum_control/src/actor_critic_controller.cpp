#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>
#include <control_msgs/msg/state_action_pair.hpp>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class simpleNN
{
public:
    simpleNN(int input_size, int hidden_size, int output_size, double learning_rate)
    {
        W1 = MatrixXd::Random(input_size, hidden_size);
        b1 = VectorXd::Zero(hidden_size);

        W2 = MatrixXd::Random(hidden_size, output_size);
        b2 = VectorXd::Zero(output_size);
    }

    double forward(const VectorXd &input)
    {
        z1 = (input.transpose() * W1).transpose() + b1;
        a1 = z1.unaryExpr([](double x)
                          { return 1.0 / (1.0 + std::exp(-x)); });

        double output = (a1.transpose() * W2).sum() + b2(0);
        return output;
    }

    void backprop(const VectorXd &input, double loss)
    {
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
    void train(const VectorXd &input, const double &desired_output)
    {
        double nn_output = forward(input);
        double loss = (nn_output - desired_output) * (nn_output - desired_output);

        backprop(input, loss);
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
    NNcontroller() : Node("nn_controller"), actor_(2, 16, 1, -0.001), critic_(2, 16, 1, 0.001)
    {
        state_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "pendulum/state", 10,
            std::bind(&NNcontroller::state_callback, this, std::placeholders::_1));

        // action_label_sub_ = this->create_subscription<control_msgs::msg::StateActionPair>(
        //     "pendulum/action_label", 10,
        //     std::bind(&NNcontroller::action_label_callback, this, std::placeholders::_1));

        nn_control_pub_ = this->create_publisher<std_msgs::msg::Float64>("pendulum/ac_nn_control", 10);
        training_data_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("pendulum/training_data", 10);

        state.resize(2);
        prev_state.resize(2);
        training_data_msg_.data.resize(3);
    }

private:
    double theta = 0.0;
    double theta_dot = 0.0;

    double nn_control = 0.0;
    VectorXd state;
    VectorXd prev_state;

    double value;
    double prev_value;
    double td_error;
    double reward;
    double pid_output;
    double gamma = 0.99;
    bool first_step = false;
    std_msgs::msg::Float64 control_msg;
    std_msgs::msg::Float64MultiArray training_data_msg_;
    double running_reward = 0.0;
    double compute_reward(double theta, double theta_dot, double action, double flag)
    {
        double r = 1 - (theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * action * action);
        if (std::abs(theta) < 0.1 )
        {
            r += 10.0; // bonus for being close to upright and still
        }

        running_reward += 0.01;
        if (flag == -1)
        {
            r -= 5.0; // or 10.0
        }
        if (flag != 0.0)
        {
            running_reward = 0.0;
        }
        return r + running_reward;
    }
    void state_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
    {
        prev_state << theta, theta_dot;
        prev_value = critic_.forward(prev_state);

        reward = compute_reward(theta, theta_dot, nn_control, msg->z);
        training_data_msg_.data[0] = reward;


        theta = msg->x;
        theta_dot = msg->y;
        state << theta, theta_dot;
        value = critic_.forward(state);
        nn_control = actor_.forward(state);

        if (msg->z == 1.0)
        {
            first_step = true;
        }
        if (msg->z == 0.0 && !first_step) // z will be 1 at episode reset
        {

            td_error = reward + gamma * value - prev_value;

            critic_.train(prev_state, reward + gamma * value);
            actor_.backprop(prev_state, td_error);
        }
        if (msg->z == 0.0)
        {
            first_step = false;
        }
        control_msg.data = nn_control;

        nn_control_pub_->publish(control_msg);
        training_data_pub_->publish(training_data_msg_);

    }

    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr state_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nn_control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr training_data_pub_;

    rclcpp::Subscription<control_msgs::msg::StateActionPair>::SharedPtr action_label_sub_;
    simpleNN actor_;
    simpleNN critic_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NNcontroller>());
    rclcpp::shutdown();
    return 0;
}
