#include "pendulum_control/actor_critic_controller.hpp"

simpleNN::simpleNN(int input_size, int hidden_size, int output_size, double learning_rate_)
{
    W1 = MatrixXd::Random(input_size, hidden_size);
    b1 = VectorXd::Zero(hidden_size);

    W2 = MatrixXd::Random(hidden_size, output_size);
    b2 = VectorXd::Zero(output_size);
    learning_rate = learning_rate_;
}

double simpleNN::forward(const VectorXd &input)
{
    z1 = (input.transpose() * W1).transpose() + b1;
    a1 = z1.unaryExpr([](double x)
                      { return 1.0 / (1.0 + std::exp(-x)); });

    double output = (a1.transpose() * W2).sum() + b2(0);
    return output;
}

void simpleNN::backprop(const VectorXd &input, double loss)
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
void simpleNN::train(const VectorXd &input, const double &desired_output)
{
    double nn_output = simpleNN::forward(input);
    // double loss = (nn_output - desired_output) * (nn_output - desired_output);
    double loss = (nn_output - desired_output);

    simpleNN::backprop(input, loss);
}


NNController::NNController(int argc, char *argv[], rclcpp::NodeOptions &options) : Node("nn_controller"), actor_(2, 16, 1, -0.001), critic_(2, 16, 1, 0.001)
{
    state_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
        "pendulum/state", 10,
        std::bind(&NNController::state_callback, this, std::placeholders::_1));

    pid_output_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "pendulum/control", 10,
        std::bind(&NNController::pid_callback, this, std::placeholders::_1));

    // action_label_sub_ = this->create_subscription<control_msgs::msg::StateActionPair>(
    //     "pendulum/acion_label", 10,
    //     std::bind(&NNController::action_label_callback, this, std::placeholders::_1));

    // action_label_sub_ = this->create_subscription<control_msgs::msg::StateActionPair>(
    //     "pendulum/action_label", 10,
    //     std::bind(&NNcontroller::action_label_callback, this, std::placeholders::_1));

    nn_control_pub_ = this->create_publisher<std_msgs::msg::Float64>("pendulum/ac_nn_control", 10);
    training_data_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("pendulum/training_data", 10);

    state.resize(2);
    prev_state.resize(2);
    training_data_msg_.data.resize(3);
    training_input.resize(2);
}

void NNController::pid_callback(const std_msgs::msg::Float64::SharedPtr msg)
{
    pid_output = msg->data;
    // actor_.train(state, pid_output);
}

void NNController::action_label_callback(const control_msgs::msg::StateActionPair::SharedPtr msg)
{

    pid_output = msg->action[0];
    training_input << msg->state[0], msg->state[1];
    actor_.train(training_input, pid_output);
}
void NNController::train_episode()
{
    training = true;
    std::vector<double> returns;
    double G = 0.0;

    // Compute discounted returns in reverse
    for (int t = reward_buffer.size() - 1; t >= 0; --t)
    {
        G = reward_buffer[t] + gamma * G;
        returns.push_back(G);
    }

    // Convert to advantage
    std::vector<double> advantages;
    for (size_t i = 0; i < returns.size(); ++i)
    {
        double value = value_buffer[i];
        double advantage = returns[i] - value;
        advantages.push_back(advantage);
    }

    for (size_t i = 0; i < state_buffer.size(); ++i)
    {
        // critic_.train(state_buffer[i], returns[i]);
        // actor_.backprop(state_buffer[i], advantages[i]);
        actor_.train(state_buffer[i], pid_output_buffer[i]);
    }

    // Clear buffers
    state_buffer.clear();
    reward_buffer.clear();
    action_buffer.clear();
    value_buffer.clear();
    pid_output_buffer.clear();
    training = false;
}

double NNController::compute_reward(double theta, double theta_dot, double action, double flag)
{
    double r = 1 - (theta * theta + 0.1 * theta_dot * theta_dot + 0.001 * action * action);
    if (std::abs(theta) < 0.1)
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
void NNController::state_callback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    if(!training){

    prev_state << theta, theta_dot;
    prev_value = critic_.forward(prev_state);

    theta = msg->x;
    theta_dot = msg->y;
    state << theta, theta_dot;
    value = critic_.forward(state);
    nn_control = actor_.forward(state);

    reward = compute_reward(theta, theta_dot, nn_control, msg->z);
    total_reward = total_reward + reward;
    training_data_msg_.data[0] = reward;
    training_data_msg_.data[1] = total_reward;

    state_buffer.push_back(state);
    reward_buffer.push_back(reward);
    action_buffer.push_back(Eigen::VectorXd::Constant(1, nn_control));
    value_buffer.push_back(value);
    pid_output_buffer.push_back(pid_output);

    if (msg->z != 0.0)
    {
        std::thread(&NNController::train_episode, this).detach();
        std::cout << "training episode " << "total reward : " << total_reward << std::endl;
        total_reward = 0.0;
    }

    control_msg.data = nn_control;

    nn_control_pub_->publish(control_msg);
    training_data_pub_->publish(training_data_msg_);
    }

}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    rclcpp::spin(std::make_shared<NNController>(argc, argv, options));
    rclcpp::shutdown();
    return 0;
}
