# lab2_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# --- Page Config ---
st.set_page_config(page_title="NN Lab 2", layout="wide", page_icon="⚡")

# --- Activation Functions & Derivatives (from your Lab 2) ---
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))
def tanh_derivative(x): return 1 - np.power(tanh(x), 2)
def relu_derivative(x): return np.where(x > 0, 1, 0)

# --- Neural Network Class (Final Corrected Version) ---
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        np.random.seed(42)
        self.input_nodes, self.hidden_nodes, self.output_nodes = input_nodes, hidden_nodes, output_nodes
        self.learning_rate = learning_rate
        self.weights_ih = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
        self.weights_ho = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))
        self.bias_h = np.random.uniform(-1, 1, (self.hidden_nodes, 1))
        self.bias_o = np.random.uniform(-1, 1, (self.output_nodes, 1))
        self.activation = 'sigmoid'
        self.loss_history = []

    def _apply_activation(self, x, derivative=False):
        if self.activation == 'sigmoid': return sigmoid_derivative(x) if derivative else sigmoid(x)
        elif self.activation == 'tanh': return tanh_derivative(x) if derivative else tanh(x)
        elif self.activation == 'relu': return relu_derivative(x) if derivative else relu(x)

    def feedforward(self, inputs):
        hidden_inputs = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden_outputs = self._apply_activation(hidden_inputs)
        final_inputs = np.dot(self.weights_ho, hidden_outputs) + self.bias_o
        final_outputs = sigmoid(final_inputs)
        return hidden_outputs, final_outputs

    def train(self, inputs_list, targets_list, activation='sigmoid'):
        self.activation = activation
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_outputs, final_outputs = self.feedforward(inputs)
        output_errors = targets - final_outputs
        gradients_o = output_errors * sigmoid_derivative(final_outputs)
        self.weights_ho += self.learning_rate * np.dot(gradients_o, hidden_outputs.T)
        self.bias_o += self.learning_rate * gradients_o
        hidden_errors = np.dot(self.weights_ho.T, output_errors)
        gradients_h = hidden_errors * self._apply_activation(hidden_outputs, derivative=True)
        self.weights_ih += self.learning_rate * np.dot(gradients_h, inputs.T)
        self.bias_h += self.learning_rate * gradients_h

    def predict(self, X_batch):
        # --- FIX IS HERE ---
        # This method is now designed to handle a batch of inputs (like X_xor).
        # It loops through each input row, gets a prediction, and collects them.
        predictions = []
        for xi in X_batch:
            # Reshape a single input row (e.g., [0, 0]) into a column vector
            inputs = np.array(xi, ndmin=2).T
            # Get the raw output from the network
            _, final_outputs = self.feedforward(inputs)
            # Apply threshold and append the result
            prediction = 1 if final_outputs[0][0] > 0.5 else 0
            predictions.append(prediction)
        # Return as a column vector to match y_xor's shape
        return np.array(predictions).reshape(-1, 1)

    def calculate_loss(self, X, y):
        loss = 0
        for i in range(len(X)):
            _, prediction = self.feedforward(np.array(X[i], ndmin=2).T)
            loss += -np.mean(y[i] * np.log(prediction + 1e-8) + (1 - y[i]) * np.log(1 - prediction + 1e-8))
        return loss / len(X)
        
# --- Plotting Functions ---
def plot_nn_boundary(X, y, model, title):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm', s=100, edgecolors='k', zorder=2)
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm', zorder=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    return fig

# --- App Layout ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>⚡ Lab 2: Activation Functions & Neural Networks</h1>", unsafe_allow_html=True)
st.sidebar.title("📄 Navigation")
page = st.sidebar.radio("Go to:", ["Activation Function Explorer", "Neural Network Sandbox", "Head-to-Head Comparison"])
X_xor, y_xor = np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([[0],[1],[1],[0]])

# --- Page 1: Activation Function Explorer ---
if page == "Activation Function Explorer":
    st.header("🔍 Activation Function Explorer")
    x = np.linspace(-10, 10, 200)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Functions")
        fig, ax = plt.subplots()
        ax.plot(x, sigmoid(x), label='Sigmoid', lw=3)
        ax.plot(x, tanh(x), label='Tanh', lw=3)
        ax.plot(x, relu(x), label='ReLU', lw=3)
        ax.set_title("Activation Functions")
        ax.set_xlabel("Input (z)"); ax.set_ylabel("Output (g(z))")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
    with col2:
        st.subheader("Derivatives")
        fig, ax = plt.subplots()
        ax.plot(x, sigmoid_derivative(x), label='Sigmoid Derivative', lw=3)
        ax.plot(x, tanh_derivative(x), label='Tanh Derivative', lw=3)
        ax.plot(x, relu_derivative(x), label='ReLU Derivative', lw=3)
        ax.set_title("Derivatives (for Backpropagation)")
        ax.set_xlabel("Input (z)"); ax.set_ylabel("Gradient")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

# --- Page 2: Neural Network Sandbox ---
elif page == "Neural Network Sandbox":
    st.header("🛠️ Neural Network Sandbox for the XOR Problem")
    with st.sidebar:
        st.header("Sandbox Controls")
        activation = st.selectbox("Activation Function:", ['sigmoid', 'tanh', 'relu'])
        hidden_nodes = st.slider("Number of Hidden Neurons:", 2, 10, 4)
        learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.1, 0.01)
        epochs = st.slider("Training Epochs:", 1000, 20000, 10000, 1000)

    if st.button("Train Network", type="primary"):
        nn = NeuralNetwork(2, hidden_nodes, 1, learning_rate)
        loss_over_time = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for e in range(epochs):
            for i in range(len(X_xor)):
                nn.train(X_xor[i], y_xor[i], activation)
            if (e + 1) % 100 == 0:
                loss = nn.calculate_loss(X_xor, y_xor)
                loss_over_time.append(loss)
                progress_bar.progress((e + 1) / epochs)
                status_text.text(f"Epoch {e+1}/{epochs} | Loss: {loss:.4f}")

        status_text.success("Training complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Loss Curve")
            st.line_chart(loss_over_time)
            
            predictions = nn.predict(X_xor)
            accuracy = accuracy_score(y_xor, predictions)
            st.metric("Final Accuracy", f"{accuracy * 100:.2f}%")
        with col2:
            st.subheader("Decision Boundary")
            fig = plot_nn_boundary(X_xor, y_xor, nn, f"{activation.upper()} Decision Boundary")
            st.pyplot(fig)

# --- Page 3: Head-to-Head Comparison ---
elif page == "Head-to-Head Comparison":
    st.header("🏆 Head-to-Head Comparison")
    epochs = st.sidebar.slider("Epochs for Comparison:", 1000, 10000, 5000)
    
    if st.button("Run Comparison", type="primary"):
        results = {}
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rates = {'sigmoid': 0.5, 'tanh': 0.5, 'relu': 0.1}
        
        for act in activations:
            with st.spinner(f"Training with {act.upper()}..."):
                nn = NeuralNetwork(2, 4, 1, learning_rates[act])
                losses = []
                for e in range(epochs):
                    for i in range(len(X_xor)):
                        nn.train(X_xor[i], y_xor[i], act)
                    if (e+1) % 50 == 0: losses.append(nn.calculate_loss(X_xor, y_xor))
                
                acc = accuracy_score(y_xor, nn.predict(X_xor))
                results[act] = {'losses': losses, 'accuracy': acc}

        st.subheader("Training Loss Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        for act, data in results.items():
            ax.plot(data['losses'], label=f"{act.upper()}", lw=3, alpha=0.8)
        ax.set_title("Loss Comparison"); ax.set_xlabel("Iterations (x50 Epochs)"); ax.set_ylabel("Loss")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)
        
        st.subheader("Final Accuracy")
        accuracies = {k: v['accuracy'] for k, v in results.items()}
        st.bar_chart(accuracies)
        
        st.success("**Key Takeaway:** While all can solve the problem, **ReLU** often converges fastest (reaches low loss in fewer epochs) but may require a lower learning rate for stability. This is why it's a popular choice in modern deep learning.")