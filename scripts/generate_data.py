import os
import numpy as np
from sklearn.datasets import load_iris

def main():
    print("Loading original Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    print("Generating synthetic 4th category...")
    np.random.seed(42)
    X_extra = X[y == 2][:30] + np.random.normal(0, 0.5, size=(30, X.shape[1]))
    y_extra = np.full(30, 3) # Class 3 is the 4th category
    
    X_combined = np.vstack((X, X_extra))
    y_combined = np.concatenate((y, y_extra))
    
    data = np.column_stack((X_combined, y_combined))
    
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/iris_dataset.csv'
    
    header = "sepal_length,sepal_width,petal_length,petal_width,target"
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
    
    print(f"Successfully saved 4-category dataset to {csv_path}")

if __name__ == "__main__":
    main()
