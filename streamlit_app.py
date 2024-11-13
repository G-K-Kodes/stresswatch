# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('SaYoPillow.csv')
df = df.rename(columns={
    'sr.1': 'Sleep Duration',
    'hr': 'Heart Rate',
    'sl': 'Stress Level',
    'sr': 'Snoring Rate',
    'rr': 'Respiration Rate',
    't': 'Body Temperature',
    'lm': 'Limb Movement',
    'bo': 'Blood Oxygen Levels',
    'rem': 'Eye Movement',
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
    fig = px.histogram(df, x="Stress Level", title="Distribution of Stress Levels")
    st.plotly_chart(fig)
    
    # Show distributions for selected features
    st.subheader("Distributions of Key Features")
    selected_features = st.multiselect("Select features to display distributions", df.columns.tolist(), 
                                       default=["Sleep Duration", "Heart Rate", "Snoring Rate", "Body Temperature"])
    for feature in selected_features:
        fig = px.histogram(df, x=feature, marginal="violin", title=f"Distribution of {feature}", nbins=30)
        st.plotly_chart(fig)

# Correlation Analysis Section
elif visualization == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values.round(2),
        x=list(corr_matrix.columns),
        y=list(corr_matrix.columns),
        colorscale="Viridis",
        showscale=True
    )
    fig.update_layout(title="Correlation Heatmap of Features", autosize=True)
    st.plotly_chart(fig)
    
    # Scatter plots for relationship with Stress Level
    st.subheader("Feature vs. Stress Level Scatter Plots")
    selected_scatter_features = st.multiselect("Select features for scatter plots", [x for x in df.columns.tolist() if x != 'Stress Level'], 
                                               default=["Sleep Duration", "Heart Rate", "Body Temperature"])
    for feature in selected_scatter_features:
        fig = px.scatter(df, x=feature, y="Stress Level", trendline="ols", title=f"{feature} vs. Stress Level")
        st.plotly_chart(fig)

# Feature Impact on Stress Level Section
elif visualization == "Feature Impact on Stress":
    st.header("Feature Analysis by Stress Level")
    
    # Box plots for feature distribution by Stress Level
    st.subheader("Box Plots of Features by Stress Level")
    selected_boxplot_features = st.multiselect("Select features for box plots", [x for x in df.columns.tolist() if x != 'Stress Level'], 
                                               default=["Snoring Rate", "Respiration Rate", "Blood Oxygen Levels"])
    for feature in selected_boxplot_features:
        fig = px.box(df, x="Stress Level", y=feature, title=f"{feature} by Stress Level")
        st.plotly_chart(fig)
    
    # Violin plots for feature distribution by Stress Level
    st.subheader("Violin Plots of Features by Stress Level")
    selected_violin_features = st.multiselect("Select features for violin plots", [x for x in df.columns.tolist() if x != 'Stress Level'], 
                                              default=["Heart Rate", "Eye Movement"])
    for feature in selected_violin_features:
        fig = px.violin(df, x="Stress Level", y=feature, box=True, points="all", title=f"{feature} by Stress Level")
        st.plotly_chart(fig)

# Predicted vs Actual Stress Levels Section
elif visualization == "Predicted vs Actual Stress Levels":
    st.header("Predicted vs Actual Stress Levels")
    X = df[['Sleep Duration', 'Heart Rate', 'Snoring Rate', 'Respiration Rate', 'Body Temperature', 'Limb Movement', 'Blood Oxygen Levels', 'Eye Movement']]
    y = df['Stress Level']

    # Train-Test Split for dataframe 1
    X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler1 = StandardScaler()
    X1_train_scaled = scaler1.fit_transform(X)

    # Train Gaussian Naive Bayes model
    model = GaussianNB(var_smoothing=1)
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
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels={"color": "Count"})
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
    st.plotly_chart(fig)

    # Scatter plot of actual vs predicted stress levels
    st.subheader("Actual vs Predicted Stress Levels Scatter Plot")
    fig = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Stress Level', 'y': 'Predicted Stress Level'}, trendline="ols")
    fig.update_layout(title="Actual vs Predicted Stress Levels")
    st.plotly_chart(fig)