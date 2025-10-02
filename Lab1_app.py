# lab1_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# --- Perceptron Class (Corrected Version) ---
class Perceptron:
    def __init__(self, n_features, learning_rate=0.1):
        np.random.seed(42)
        self.lr = learning_rate
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()
        # FIX: Initialize history attribute here to ensure it always exists
        self.history = {'errors': [], 'weights': [], 'bias': []}
    
    def _step_activation(self, z):
        return 1 if z >= 0 else 0
    
    def predict_single(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self._step_activation(z)

    def fit(self, X, y, epochs=100):
        # Reset history for each new training run
        self.history = {'errors': [], 'weights': [], 'bias': []}
        for epoch in range(epochs):
            error_count = 0
            for xi, target in zip(X, y):
                prediction = self.predict_single(xi)
                error = target - prediction
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_count += 1
            self.history['errors'].append(error_count)
            self.history['weights'].append(self.weights.copy())
            self.history['bias'].append(self.bias)
            if error_count == 0:
                break
        return self

# --- Helper Functions for Plotting ---
def plot_decision_boundary(X, y, model, title):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100, edgecolors='k')
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = np.array([model.predict_single(np.array([x1, x2])) for x1, x2 in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Input A")
    ax.set_ylabel("Input B")
    ax.grid(True, alpha=0.3)
    return fig

def plot_errors(errors, title):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', color='b')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Number of Misclassifications")
    ax.grid(True, alpha=0.3)
    return fig

# --- Streamlit App UI ---
st.set_page_config(page_title="Perceptron Lab 1", layout="wide", page_icon="💡")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>💡 Lab 1: Single Layer Perceptron for Logic Gates</h1>", unsafe_allow_html=True)
st.write("An interactive app to train and visualize a Perceptron learning basic, linearly separable logic gates.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    gate = st.selectbox("Select a Logic Gate:", ["AND", "OR", "AND-NOT", "XOR"])
    learning_rate = st.slider("Learning Rate (η):", 0.01, 1.0, 0.1, 0.01)
    epochs = st.slider("Max Epochs:", 10, 500, 100, 10)
    train_button = st.button("🚀 Train Model", type="primary")

# --- Datasets ---
datasets = {
    "AND": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,0,1])),
    "OR": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,1])),
    "AND-NOT": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,0,1,0])),
    "XOR": (np.array([[0,0],[0,1],[1,0],[1,1]]), np.array([0,1,1,0]))
}
X, y = datasets[gate]

# --- Main App Logic ---
if 'model' not in st.session_state or train_button:
    st.session_state.model = Perceptron(n_features=2, learning_rate=learning_rate)
    st.session_state.errors = []
    
if train_button:
    with st.spinner(f"Training on {gate} gate..."):
        st.session_state.model.fit(X, y, epochs)
        time.sleep(1) # For dramatic effect
    st.success(f"Training complete! Converged in {len(st.session_state.model.history['errors'])} epochs.")

model = st.session_state.model

# --- Display Area using Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Training Visualization", "🧪 Interactive Testbed", "🧠 Concepts & Answers"])

with tab1:
    st.header("Training Performance")
    if not model.history['errors']:
        st.info("Click the 'Train Model' button in the sidebar to start.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Decision Boundary")
            if gate == "XOR" and model.history['errors'][-1] != 0:
                 st.error("XOR is not linearly separable. The Perceptron could not find a line to separate the classes.", icon="❌")
            fig_boundary = plot_decision_boundary(X, y, model, f"{gate} Gate Decision Boundary")
            st.pyplot(fig_boundary)
        with col2:
            st.subheader("Misclassifications per Epoch")
            fig_errors = plot_errors(model.history['errors'], "Convergence Plot")
            st.pyplot(fig_errors)
        
        st.metric("Final Bias (b)", f"{model.bias:.4f}")
        st.metric("Final Weights (w1, w2)", f"[{model.weights[0]:.4f}, {model.weights[1]:.4f}]")

with tab2:
    st.header("Test the Trained Model")
    if not model.history['errors']:
        st.info("Train a model first to use the testbed.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select Inputs")
            input_a = st.toggle("Input A", value=False)
            input_b = st.toggle("Input B", value=False)
            test_input = np.array([1 if input_a else 0, 1 if input_b else 0])
        
        with col2:
            st.subheader("Prediction")
            prediction = model.predict_single(test_input)
            st.markdown(f"## Output: `{prediction}`")
            
        st.subheader("Full Verification Table")
        predictions = [model.predict_single(xi) for xi in X]
        df = pd.DataFrame({
            'Input A': X[:, 0], 'Input B': X[:, 1], 'Expected': y, 'Predicted': predictions
        })
        st.dataframe(df, use_container_width=True)

with tab3:
    st.header("Key Concepts from Lab 1")
    st.subheader("Why does the Perceptron fail on the XOR gate?")
    st.error("**Linear Separability:** The Perceptron can only solve problems where a single straight line can separate the two classes (0s and 1s).")
    st.write("- **AND, OR, AND-NOT** are all linearly separable.")
    st.write("- **XOR** is not. The points are arranged in a diagonal pattern that cannot be separated by one line. This failure was a historic moment in AI, leading to the development of multi-layer networks.")
    
    st.subheader("How do weights and bias change?")
    st.info("**The Perceptron Learning Rule:** `weight = weight + learning_rate * error * input`")
    st.write("When the model makes a mistake (`error != 0`), it nudges the weights and bias in a direction that would make that prediction correct. This process repeats until no more mistakes are made, effectively 'drawing' the decision boundary line.")