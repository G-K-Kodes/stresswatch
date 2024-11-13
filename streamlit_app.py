# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
                                     ["Data Overview", "Correlation Analysis", "Feature Impact on Stress"])

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
