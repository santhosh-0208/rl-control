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
    simpleNN();
    simpleNN(int input_size, int hidden_size, int output_size, double learning_rate);
    double forward(const VectorXd &input);
    void backprop(const VectorXd &input, double loss);
    void train(const VectorXd &input, const double &desired_output);
private:
    MatrixXd W1;
    MatrixXd W2;
    VectorXd b1;
    VectorXd b2;

    MatrixXd z1;
    MatrixXd a1;

    double learning_rate;
};

class NNController : public rclcpp::Node
{
public:
    NNController(int argc, char *argv[], rclcpp::NodeOptions &options);
    NNController();
private:
    void state_callback(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void pid_callback(const std_msgs::msg::Float64::SharedPtr msg);
    void action_label_callback(const control_msgs::msg::StateActionPair::SharedPtr msg);
    void train_episode();

    double compute_reward(double theta, double theta_dot, double action, double flag);

    std::vector<Eigen::VectorXd> state_buffer;
    std::vector<double> reward_buffer;
    std::vector<Eigen::VectorXd> action_buffer;
    std::vector<double> value_buffer;
    std::vector<double> pid_output_buffer;
    double pid_output;


    Eigen::VectorXd state;
    double nn_control;
    bool episode_done = false;
    double gamma = 0.99;

    simpleNN actor_;
    simpleNN critic_;

    double theta = 0.0;
    double theta_dot = 0.0;

    VectorXd prev_state;

    double value;
    double prev_value;
    double td_error;
    double reward;
    double total_reward;

    bool first_step = false;
    bool training = false;
    std_msgs::msg::Float64 control_msg;
    std_msgs::msg::Float64MultiArray training_data_msg_;
    double running_reward = 0.0;
    Eigen::VectorXd training_input;


    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr state_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr nn_control_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr training_data_pub_;

    rclcpp::Subscription<control_msgs::msg::StateActionPair>::SharedPtr action_label_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr pid_output_sub_;


};
