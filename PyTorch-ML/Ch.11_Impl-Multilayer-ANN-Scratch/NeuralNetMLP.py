import numpy as np


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_features = num_features

        rng = np.random.RandomState(random_seed)
        self.w1 = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.b1 = np.zeros(num_hidden)

        self.w2 = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.b2 = np.zeros(num_classes)

    def forward(self, x):
        z_h = np.dot(x, self.w1.T) + self.b1
        a_h = np.sigmoid(z_h)

        z_out = np.dot(a_h, self.w2.T) + self.b2
        a_out = np.sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        y_temp = np.zeros(y.shape[0], self.num_classes)
        for i, v in enumerate(y):
            y_temp[i, v] = 1

        y_onehot = y_temp  # One-hot encoding for multiplication during backpropagation. For shape consistency.

        dL_by_dAout = 2.0 * (a_out - y_onehot) / y.shape[0]  # Loss derivative
        dAout_by_dZout = a_out * (1.0 - a_out)  # Signmoid derivative

        delta_out = dL_by_dAout * dAout_by_dZout  # Delta error for output layer.

        dZout_by_dW2 = a_h  # Derivative of Zout w.r.t. W2

        dL_by_dW2 = np.dot(
            delta_out.T, dZout_by_dW2
        )  # Derivative of loss w.r.t. weight W2.
        dL_by_dB2 = np.sum(delta_out, axis=0)  # Derivative of loss w.r.t. bias B2.

        dZout_by_Ah = self.w2  # Derivative of Zout w.r.t. hidden activation Ah.
        dL_by_Ah = np.dot(
            delta_out, dZout_by_Ah
        )  # Derivative of loss w.r.t. hidden activation Ah.

        dAh_by_dZh = a_h * (
            1.0 - a_h
        )  # Derivative of hidden activation Ah w.r.t. hidden layer input Zh.
        dZh_by_dWh = x  # Derivative of hidden layer input Zh w.r.t. weight Wh.

        dL_by_dWh = np.dot(
            (dL_by_Ah * dAh_by_dZh).T, dZh_by_dWh
        )  # Derivative of loss w.r.t. weight Wh. (??)
        dL_by_dBh = np.sum(
            (dL_by_Ah * dAh_by_dZh), axis=0
        )  # Derivative of loss w.r.t. bias Bh.

        return dL_by_dW2, dL_by_dB2, dL_by_dWh, dL_by_dBh


## Example usage.
# model_mnist = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)
