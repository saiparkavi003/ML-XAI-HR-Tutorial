import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import shap

# 1. Loading Data
# We are using the IBM HR Attrition dataset from our 'data' folder
print("Loading HR Attrition Data...")
df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# 2. Cleaning & Preparing the data
# a. First, we turn 'Attrition' (Yes/No) into numbers (1/0)
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# b. We remove columns that are useless for prediction (like Employee IDs)
# These columns don't have any patterns the model can learn from.
cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(cols_to_drop, axis=1)

# c. We turn text categories (like JobRole and Department) into numbers
# We use 'dtype=int' to make sure the library doesn't crash on a Mac.
df_final = pd.get_dummies(df, dtype=int)

# 3. Spliting the Data
# We separate our 'Target' (who left) from our 'Features' (everything else)
X = df_final.drop('Attrition', axis=1)
y = df_final['Attrition']

# We split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Traning The Model
# We use a Random Forest. I added 'balanced' weights because most employees 
# in the data actually stayed, and we want the model to pay attention to those who left.
model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
print("Training the Random Forest model...")
model.fit(X_train, y_train)

# Checking how good our model is using F1-Score
preds = model.predict(X_test)
print(f"Model Training Done! F1-Score: {f1_score(y_test, preds):.4f}")

# 5. Breaking The 'Black Box' With SHAP ---
# Now we use SHAP (Game Theory) to explain what the model learned.
print("Calculating SHAP values... this is where we explain the decisions.")
explainer = shap.TreeExplainer(model)
shap_values_raw = explainer.shap_values(X_test)

# This part is a bit technical: we are making sure we only look at 
# the reasons why people LEAVE (Class 1) to keep our plots clear.
if isinstance(shap_values_raw, list):
    plot_values = np.array(shap_values_raw[1])
    expected_val = float(explainer.expected_value[1])
else:
    # If it's a 3D array, we just slice it to get Class 1
    if len(shap_values_raw.shape) == 3:
        plot_values = shap_values_raw[:, :, 1]
        expected_val = float(explainer.expected_value[1])
    else:
        plot_values = shap_values_raw
        expected_val = float(explainer.expected_value)

# 6. Global View (Why do people leave in general?) ---
print("Generating the Beeswarm Plot...")

# We wrap everything into an 'Explanation' object so the plot looks professional
explanation = shap.Explanation(
    values=plot_values, 
    base_values=expected_val, 
    data=X_test, 
    feature_names=X_test.columns
)

plt.figure(figsize=(10, 8))
# This plot shows the top 15 reasons why people quit
shap.plots.beeswarm(explanation, max_display=15, show=False)

plt.title("Explainable AI: Key Drivers of Employee Attrition", fontsize=16, pad=25)
plt.savefig('perfect_beeswarm_plot.png', bbox_inches='tight', dpi=300)
print("Success! Beeswarm plot saved as perfect_beeswarm_plot.png")
plt.show()

# 7. Individual View (Why did THIS specific person leave?) ---
print("Generating the Waterfall Plot for one employee...")
plt.figure()

# We pick the very first person in our test set to explain their specific case
shap.plots.waterfall(explanation[0], show=False)

plt.title("Individual Explanation: Why did this employee leave?", fontsize=12, pad=20)
plt.savefig('individual_explanation.png', bbox_inches='tight', dpi=300)
print("Success! Individual plot saved as individual_explanation.png")