# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv('SaYoPillow.csv')
df = df.rename(columns={
    'sr.1': 'Sleep Duration',
    'hr': 'Heart Rate',
    'sl': 'Stress Level',
    'sr':'Snoring Rate',
    'rr':'Respiration Rate',
    't':'Body Temperature',
    'lm':'Limb Movement',
    'bo':'Blood Oxygen Levels',
    'rem':'Eye Movement',
    })

# Title and Description
st.title("Stress Level Analysis Dashboard")
st.write("""
This dashboard provides insights into how various factors such as sleep, respiration, and body metrics relate to stress levels.
""")

# Sidebar for selecting visualizations
st.sidebar.title("Navigation")
visualization = st.sidebar.selectbox("Select a Visualization", 
                                     ["Data Overview", "Correlation Analysis", "Feature Impact on Stress", "Predicted vs Actual Stress Levels"])

# Data Overview Section
if visualization == "Data Overview":
    st.header("Data Overview")
    
    # Show basic statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    # Display stress level distribution
    st.subheader("Stress Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Stress Level', ax=ax)
    ax.set_title("Distribution of Stress Levels")
    st.pyplot(fig)
    
    # Show distributions for selected features
    st.subheader("Distributions of Key Features")
    selected_features = st.multiselect("Select features to display distributions", df.columns.tolist(), 
                                       default=["Sleep Duration", "Heart Rate", "Snoring Rate", "Body Temperature"])
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

# Correlation Analysis Section
elif visualization == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Features")
    st.pyplot(fig)
    
    # Scatter plots for relationship with Stress Level
    st.subheader("Feature vs. Stress Level Scatter Plots")
    selected_scatter_features = st.multiselect("Select features for scatter plots", [x for x in df.columns.tolist() if x != 'Stress Level'], 
                                               default=["Sleep Duration", "Heart Rate", "Body Temperature"])
    for feature in selected_scatter_features:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=feature, y="Stress Level", ax=ax)
        ax.set_title(f"{feature} vs. Stress Level")
        st.pyplot(fig)

# Feature Impact on Stress Level Section
elif visualization == "Feature Impact on Stress":
    st.header("Feature Analysis by Stress Level")
    
    # Box plots for feature distribution by Stress Level
    st.subheader("Box Plots of Features by Stress Level")
    selected_boxplot_features = st.multiselect("Select features for box plots",[x for x in df.columns.tolist() if x != 'Stress Level'], 
                                               default=["Snoring Rate", "Respiration Rate", "Blood Oxygen Levels"])
    for feature in selected_boxplot_features:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Stress Level", y=feature, ax=ax)
        ax.set_title(f"{feature} by Stress Level")
        st.pyplot(fig)
    
    # Violin plots for feature distribution by Stress Level
    st.subheader("Violin Plots of Features by Stress Level")
    selected_violin_features = st.multiselect("Select features for violin plots", [x for x in df.columns.tolist() if x != 'Stress Level'], 
                                              default=["Heart Rate", "Eye Movement"])
    for feature in selected_violin_features:
        fig, ax = plt.subplots()
        sns.violinplot(data=df, x="Stress Level", y=feature, ax=ax)
        ax.set_title(f"{feature} by Stress Level")
        st.pyplot(fig)

elif visualization == "Predicted vs Actual Stress Levels":
    st.header("Predicted vs Actual Stress Levels")
    X = df[['Sleep Duration', 'Heart Rate', 'Snoring Rate', 'Respiration Rate', 'Body Temperature', 'Limb Movement', 'Blood Oxygen Levels','Eye Movement']]
    y = df['Stress Level']

    # Train-Test Split for dataframe 1
    X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler1 = StandardScaler()
    X1_train_scaled = scaler1.fit_transform(X)

    # Train Gaussian Naive Bayes model
    model = GaussianNB(var_smoothing = 1)
    model.fit(X, y)

    # Predictions and Evaluation for dataframe 1
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    st.subheader(f"Model Accuracy: {accuracy:.2f}")
    
    # Display a classification report
    st.subheader("Classification Report")
    report = classification_report(y, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Confusion matrix heatmap
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # Scatter plot of actual vs predicted stress levels
    st.subheader("Actual vs Predicted Stress Levels Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=y_pred, ax=ax)
    ax.set_xlabel("Actual Stress Level")
    ax.set_ylabel("Predicted Stress Level")
    ax.set_title("Actual vs Predicted Stress Levels")
    st.pyplot(fig)
