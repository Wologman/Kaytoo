from sklearn.model_selection import BaseCrossValidator
import numpy as np
from collections import Counter

class CustomStratifiedKFold(BaseCrossValidator):
    def __init__(self, max_class_size=None, n_splits=5):
        self.max_class_size = max_class_size
        self.n_splits = n_splits
    
    def _split(self, X, y):
        # Ensure we have the same number of samples in each fold
        class_counts = Counter(y)
        min_class_size = min(class_counts.values())
        
        # Calculate the number of samples for each class per fold
        max_class_size = self.max_class_size if self.max_class_size is not None else min_class_size
        per_class_counts = {cls: min(max_class_size, count // self.n_splits) for cls, count in class_counts.items()}
        
        # Create folds
        indices = np.arange(len(y))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        folds = [[] for _ in range(self.n_splits)]
        class_indices = {cls: np.where(y == cls)[0] for cls in class_counts}
        
        for cls, indices in class_indices.items():
            np.random.shuffle(indices)
            for fold_index in range(self.n_splits):
                start = fold_index * per_class_counts[cls]
                end = start + per_class_counts[cls]
                folds[fold_index].extend(indices[start:end])
        
        # Create training and validation indices
        for i in range(self.n_splits):
            val_indices = folds[i]
            train_indices = np.concatenate([folds[j] for j in range(self.n_splits) if j != i])
            yield train_indices, val_indices
    
    def split(self, X, y=None):
        for train_index, val_index in self._split(X, y):
            yield train_index, val_index
    
    def get_n_splits(self, X=None, y=None):
        return self.n_splits

# Example usage
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Features
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Target labels

max_class_size = 2  # Example constraint
cv = CustomStratifiedKFold(max_class_size=max_class_size, n_splits=3)

for train_index, val_index in cv.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    print(f"Train indices: {train_index}, Validation indices: {val_index}")
    print(f"X_train: {X_train}, X_val: {X_val}")
    print(f"y_train: {y_train}, y_val: {y_val}")