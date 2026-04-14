import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    # 1. Load Iris Dataset with 4 categories from CSV
    print("Loading dataset from data/iris_dataset.csv...")
    data = np.loadtxt('data/iris_dataset.csv', delimiter=',', skiprows=1)
    X, y = data[:, :-1], data[:, -1].astype(int)
    
    # 2. Split data: 80% Training, 20% Testing
    print("Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Enhance features with Polynomials to find non-linear separation boundaries
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)
    
    # Scale features to improve gradient descent convergence
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 3. Implement Classification Algorithm using MSE loss to find weights (w)
    print("Training model using MSE loss...")
    
    # Helper to one-hot encode targets for MSE calculation (N x 4)
    def one_hot(labels, num_classes=4):
        return np.eye(num_classes)[labels]
    
    Y_train = one_hot(y_train, 4)
    
    # Add bias term (column of 1s) to features
    X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    
    # Initialize weights w (shape: features+1 x 4 classes)
    w = np.zeros((X_train_b.shape[1], 4))
    
    learning_rate = 0.05
    epochs = 40000
    mse_history = []
    
    # Optimizer: Gradient Descent with Momentum
    momentum = 0.9
    v = np.zeros_like(w)
    
    # Gradient Descent loop minimizing Mean Squared Error (MSE)
    for epoch in range(epochs):
        # Predictions
        predictions = X_train_b.dot(w)
        
        # Calculate Error
        error = predictions - Y_train
        
        # Calculate MSE
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        
        # Calculate Gradient
        gradient = (2 / X_train_b.shape[0]) * X_train_b.T.dot(error)
        
        # Update weights using momentum
        v = momentum * v + learning_rate * gradient
        w -= v

    # 4. Testing and Output generation
    print("Evaluating model...")
    test_predictions = X_test_b.dot(w)
    # The predicted class is the one with the highest score
    y_pred = np.argmax(test_predictions, axis=1)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # 5. Generate PNG Reports
    print("Generating PNG reports...")
    
    # Convergence Graph
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(mse_history, color='blue', linewidth=2)
    plt.title('Convergence Graph (MSE over Iterations)')
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.savefig('convergence_graph.png')
    plt.close()
    
    # Confusion Matrix
    fig3, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig3.colorbar(cax)
    
    # Add numbers to the matrix blocks
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center',
                    color="white" if cm[i, j] > cm.max()/2. else "black")
            
    plt.title('Confusion Matrix (4x4)', pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("Done!")

if __name__ == "__main__":
    main()
