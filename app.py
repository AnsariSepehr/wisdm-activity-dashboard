import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title and Description
st.title("WISDM Activity Recognition Dashboard")
st.markdown("""
This dashboard visualizes your WISDM project. It loads the processed data, trains an XGBoost model, 
and shows key visualizations like confusion matrix, feature importance, and box plots.
""")

# Step 1: Load Data (adapt paths if your CSVs are in a subfolder)
@st.cache_data  # Caches data loading for speed
def load_data():
    X = pd.read_csv("X_final_scaled_windows.csv")
    y = pd.read_csv("y_final_labels.csv")['Encoded_Label']  # Assuming 'Encoded_Label' is the target
    pca_df = pd.read_csv("pca_reduced_dataset.csv")
    return X, y, pca_df

X, y, pca_df = load_data()

# Display Raw Data Tables (Interactive)
st.header("Data Overview")
st.subheader("Scaled Features (X_final_scaled_windows.csv)")
st.dataframe(X.head(10))  # Show first 10 rows

st.subheader("Labels (y_final_labels.csv)")
st.dataframe(pd.read_csv("y_final_labels.csv").head(10))

st.subheader("PCA Reduced Data (pca_reduced_dataset.csv)")
st.dataframe(pca_df.head(10))

# Interactive Filter: Select Activity
activities = pca_df['Activity_Label'].unique() if 'Activity_Label' in pca_df.columns else []
selected_activity = st.selectbox("Filter by Activity", ["All"] + list(activities))
if selected_activity != "All":
    filtered_pca = pca_df[pca_df['Activity_Label'] == selected_activity]
    st.dataframe(filtered_pca)
else:
    filtered_pca = pca_df

# Step 2: Train XGBoost Model (from your notebook)
st.header("Model Training and Evaluation")
le = LabelEncoder()
y_encoded = le.fit_transform(pd.read_csv("y_final_labels.csv")['Activity_Name'])  # Use Activity_Name for encoding

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, objective='multi:softprob', num_class=len(le.classes_),
    n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {acc:.2%}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Visualization 1: Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title('Confusion Matrix')
ax.set_ylabel('True Activity')
ax.set_xlabel('Predicted Activity')
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualization 2: Feature Importance
st.subheader("Top 15 Feature Importance")
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(15)

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis', ax=ax2)
ax2.set_title('Top 15 Most Important Features')
ax2.set_xlabel('Importance Score (XGBoost)')
ax2.grid(axis='x', alpha=0.3)
st.pyplot(fig2)

# Visualization 3: Box Plot of Top Feature
st.subheader("Box Plot of Top Feature by Activity")
top_feature = feature_imp.iloc[0]['Feature']
plot_data = pd.DataFrame({
    'Activity': le.inverse_transform(y_train),
    'Value': X_train[top_feature].values
})

fig3, ax3 = plt.subplots(figsize=(14, 6))
sns.boxplot(x='Activity', y='Value', data=plot_data, palette='tab20', showfliers=False, ax=ax3)
ax3.set_title(f'Distribution of Top Feature "{top_feature}" by Activity')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
ax3.grid(axis='y', alpha=0.3)
st.pyplot(fig3)

# Optional: PCA Scatter Plot (Interactive with Plotly)
st.subheader("PCA Reduced Data Scatter Plot")
import plotly.express as px
fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Activity_Label', title="PCA Visualization")
st.plotly_chart(fig_pca)