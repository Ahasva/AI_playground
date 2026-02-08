/*
 * Simple Neural Network to learn y = 2x
 */

#include <iostream>
#include <vector>

// -----------------------------
// Activation: ReLU
// -----------------------------
double relu(double x) {
    return (x > 0.0) ? x : 0.0;
}

double relu_derivative(double x) {
    return (x > 0.0) ? 1.0 : 0.0;
}

// -----------------------------
// Simple 1-2-1 Neural Network
// Input: 1 neuron
// Hidden: 2 neurons (ReLU)
// Output: 1 neuron (linear)
// -----------------------------
struct Net {
    // input -> hidden
    double w1, b1; // to hidden neuron 1
    double w2, b2; // to hidden neuron 2

    // hidden -> output
    double w3;     // from hidden neuron 1 to output
    double w4;     // from hidden neuron 2 to output
    double b3;     // output bias
};

struct ForwardCache {
    double x;

    double z1, z2; // pre-activation hidden
    double h1, h2; // post-activation hidden

    double y_pred; // output
};

// -----------------------------
// Initialize network parameters
// -----------------------------
Net init_net() {
    Net net;
    net.w1 = 0.5;  net.b1 = 0.0;
    net.w2 = -0.5; net.b2 = 0.0;
    net.w3 = 0.5;
    net.w4 = 0.5;
    net.b3 = 0.0;
    return net;
}

// -----------------------------
// Forward pass: compute prediction
// -----------------------------
ForwardCache forward(const Net& net, double x) {
    ForwardCache c;
    c.x = x;

    // hidden layer pre-activations
    c.z1 = net.w1 * x + net.b1;
    c.z2 = net.w2 * x + net.b2;

    // hidden layer activations
    c.h1 = relu(c.z1);
    c.h2 = relu(c.z2);

    // output (linear)
    c.y_pred = net.w3 * c.h1 + net.w4 * c.h2 + net.b3;
    return c;
}

// -----------------------------
// Loss: Mean Squared Error for one sample
// -----------------------------
double mse_loss(double y_true, double y_pred) {
    double e = y_true - y_pred;
    return e * e;
}

// -----------------------------
// Backward pass + parameter update (SGD)
// -----------------------------
void backward_and_update(Net& net,
                         const ForwardCache& c,
                         double y_true,
                         double learning_rate)
{
    // Loss: L = (y - y_pred)^2
    // dL/dy_pred = -2 (y - y_pred)
    double dL_dy = -2.0 * (y_true - c.y_pred);

    // Output layer:
    // y_pred = w3*h1 + w4*h2 + b3
    double dL_dw3 = dL_dy * c.h1;
    double dL_dw4 = dL_dy * c.h2;
    double dL_db3 = dL_dy;

    // Gradients wrt hidden activations
    double dL_dh1 = dL_dy * net.w3;
    double dL_dh2 = dL_dy * net.w4;

    // Hidden layer:
    // h = ReLU(z), z = w*x + b
    double dL_dz1 = dL_dh1 * relu_derivative(c.z1);
    double dL_dz2 = dL_dh2 * relu_derivative(c.z2);

    double dL_dw1 = dL_dz1 * c.x;
    double dL_db1 = dL_dz1;

    double dL_dw2 = dL_dz2 * c.x;
    double dL_db2 = dL_dz2;

    // Gradient descent update: param -= lr * grad
    net.w3 -= learning_rate * dL_dw3;
    net.w4 -= learning_rate * dL_dw4;
    net.b3 -= learning_rate * dL_db3;

    net.w1 -= learning_rate * dL_dw1;
    net.b1 -= learning_rate * dL_db1;

    net.w2 -= learning_rate * dL_dw2;
    net.b2 -= learning_rate * dL_db2;
}

// -----------------------------
// Training loop
// -----------------------------
void train(Net& net,
           const std::vector<double>& x_data,
           const std::vector<double>& y_data,
           int epochs,
           double learning_rate)
{
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < x_data.size(); ++i) {
            // 1) forward
            ForwardCache c = forward(net, x_data[i]);

            // 2) loss
            double loss = mse_loss(y_data[i], c.y_pred);
            total_loss += loss;

            // 3) backward + update
            backward_and_update(net, c, y_data[i], learning_rate);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << (total_loss / x_data.size())
                      << std::endl;
        }
    }
}

double predict(const Net& net, double x) {
    return forward(net, x).y_pred;
}

int main() {
    // Data: y = 2x
    std::vector<double> x_data = {1, 2, 3, 4};
    std::vector<double> y_data = {2, 4, 6, 8};

    Net net = init_net();

    int epochs = 1000;
    double lr = 0.01;

    train(net, x_data, y_data, epochs, lr);

    // Test prediction
    double x_test = 5.0;
    double y_hat = predict(net, x_test);

    std::cout << "\nTest input: " << x_test
              << " | Prediction: " << y_hat
              << std::endl;

    return 0;
}
