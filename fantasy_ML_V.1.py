import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example data loading
# Generating dummy data for the quarterback features based on the relevant fields
np.random.seed(42)
data = pd.DataFrame({
    'passing_attempts': np.random.randint(0, 80),
    'passing_completions': np.random.randint(0, 80),
    'passing_yards': np.random.randint(0, 600),
    'touchdowns': np.random.randint(0, 10),
    'interceptions': np.random.randint(0, 10),
    'rushing_yards': np.random.randint(0, 400),
    'completion_percentage': np.random.uniform(0, 100),
    'passer_rating': np.random.uniform(0, 158.3),
    'opponent_defense_rank': np.random.randint(1, 32),
    'time_of_possession': np.random.uniform(0, 50),  # Minutes
    'aggressiveness': np.random.uniform(0, 50), #percentage
    'avg_time_to_throw': np.random.uniform(2, 4, 1000),  # Seconds
    'actual_points': np.random.uniform(0, 100), #fantasy points
    'projected_points': np.random.uniform(0, 30), #fantasy points
    })

# Create the target variable (1 = Boom, 0 = Bust)
data['boom_or_bust'] = np.where(data['actual_points'] > data['projected_points'], 1, 0)

# Features and target
X = data[['passing_attempts', 'passing_completions', 'passing_yards', 'touchdowns', 
          'interceptions', 'rushing_yards', 'completion_percentage', 
          'passer_rating', 'opponent_defense_rank', 'time_of_possession', 
          'aggressiveness', 'avg_time_to_throw', 'projected_points']]
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
