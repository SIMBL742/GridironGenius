import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example data loading
# Assume the dataset contains columns like: player_name, position, projections, actual_points, etc.
# For simplicity, we'll generate dummy data.
np.random.seed(42)
data = pd.DataFrame({
    'rushing_yards': np.random.randint(0, 150, 1000),
    'receiving_yards': np.random.randint(0, 200, 1000),
    'touchdowns': np.random.randint(0, 4, 1000),
    'targets': np.random.randint(0, 15, 1000),
    'opponent_defense_rank': np.random.randint(1, 32, 1000),
    'actual_points': np.random.uniform(0, 40, 1000),
    'projected_points': np.random.uniform(0, 40, 1000),
})

# Create the target variable (1 = Boom, 0 = Bust)
data['boom_or_bust'] = np.where(data['actual_points'] > data['projected_points'], 1, 0)

# Features and target
X = data[['rushing_yards', 'receiving_yards', 'touchdowns', 'targets', 'opponent_defense_rank', 'projected_points']]
y = data['boom_or_bust']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)
