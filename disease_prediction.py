# disease_prediction.py
# A Beginner-Friendly Machine Learning Project for Disease Prediction

# ==============================
# Step 1: Import Required Libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

# Machine Learning Tools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, confusion_matrix

# Machine Learning Models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ==============================
# Step 2: Load and Prepare Data
# ==============================
# Load the dataset (save CSV file in same folder as this script)
# Download link: [Insert your dataset link here]
data = pd.read_csv('improved_disease_dataset.csv')

# Show first 5 rows
print("\nFirst look at the data:")
print(data.head())

# ==============================
# Step 3: Clean and Prepare Data
# ==============================
# Convert disease names to numbers
label_encoder = LabelEncoder()
data["disease"] = label_encoder.fit_transform(data["disease"])

# Handle categorical data (like gender if present)
if 'gender' in data.columns:
    data['gender'] = LabelEncoder().fit_transform(data['gender'])

# Split data into features (X) and target (y)
X = data.drop('disease', axis=1)  # All columns except disease
y = data['disease']               # Disease column

# ==============================
# Step 4: Handle Class Imbalance
# ==============================
# Balance the dataset
balancer = RandomOverSampler(random_state=42)
X_balanced, y_balanced = balancer.fit_resample(X, y)

# Show class distribution
plt.figure(figsize=(15, 6))
sns.countplot(x=y_balanced)
plt.title("Disease Distribution After Balancing")
plt.xticks(rotation=90)
plt.show()

# ==============================
# Step 5: Train Machine Learning Models
# ==============================
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.2,
    random_state=42
)

# Initialize models
models = {
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Show results
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# ==============================
# Step 6: Combined Prediction System
# ==============================
# Train final models on full data
final_svm = SVC().fit(X_balanced, y_balanced)
final_nb = GaussianNB().fit(X_balanced, y_balanced)
final_rf = RandomForestClassifier().fit(X_balanced, y_balanced)

def predict_disease(symptoms: str) -> dict:
    """
    Predicts disease based on symptoms
    Input format: "symptom1, symptom2, symptom3"
    Example: "itching, skin_rash, nodal_skin_eruptions"
    """
    # Create empty symptom vector
    symptom_vector = np.zeros(len(X.columns))
    
    # Convert input symptoms to vector
    for symptom in symptoms.lower().split(','):
        symptom = symptom.strip()
        if symptom in X.columns:
            index = list(X.columns).index(symptom)
            symptom_vector[index] = 1
    
    # Make predictions
    svm_pred = final_svm.predict([symptom_vector])[0]
    nb_pred = final_nb.predict([symptom_vector])[0]
    rf_pred = final_rf.predict([symptom_vector])[0]
    
    # Combine predictions
    final_pred = mode([svm_pred, nb_pred, rf_pred])[0][0]
    
    return {
        "Symptoms Input": symptoms,
        "SVM Prediction": label_encoder.inverse_transform([svm_pred])[0],
        "Naive Bayes Prediction": label_encoder.inverse_transform([nb_pred])[0],
        "Random Forest Prediction": label_encoder.inverse_transform([rf_pred])[0],
        "Final Prediction": label_encoder.inverse_transform([final_pred])[0]
    }

# ==============================
# Step 7: Test the System
# ==============================
# Example test (use actual symptoms from your dataset)
test_case = "itching, skin_rash, nodal_skin_eruptions"
print("\nTest Prediction:")
print(predict_disease(test_case))

# ==============================
# How to Use This Program:
# ==============================
"""
1. Save this file as 'disease_prediction.py'
2. Place your dataset CSV file in the same folder
3. Install requirements:
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn scipy
4. Run the script:
   python disease_prediction.py
5. Modify the test_case variable to try different symptoms
"""