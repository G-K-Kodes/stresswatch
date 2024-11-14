# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from scipy.stats import f_oneway, kstest, shapiro
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

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
                                     ["Data Overview", 
                                      "Correlation Analysis", 
                                      "Feature Impact on Stress", 
                                      "Predicted vs Actual Stress Levels", 
                                      "Hypothesis Testing",
                                      "Normality Testing"])

# Data Overview Section
if visualization == "Data Overview":
    st.header("Data Overview")
    
    # Show basic statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())
    
    # Display stress level distribution
    st.subheader("Stress Level Distribution")
    fig = px.histogram(df, x="Stress Level", title="Distribution of Stress Levels", nbins = 30)
    st.plotly_chart(fig)
    
    # Show distributions for selected features
    st.subheader("Distributions of Key Features")
    selected_features = st.multiselect("Select features to display distributions", [x for x in df.columns.tolist() if x != 'Stress Level'], 
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

elif visualization == "Hypothesis Testing":
    st.header("Hypothesis Testing")
    
    # Feature Selection for Hypothesis Testing
    selected_features = st.multiselect(
        "Select features for Hypothesis Testing",df.columns.tolist(),
        default=["Sleep Duration", "Stress Level"]
    )
    
    # If exactly two features are selected, perform a two-sample Z-test
    if len(selected_features) == 2:
        st.subheader("Two-Sample Z-Test")
        
        # Extract data for the selected features
        data1 = df[selected_features[0]]
        data2 = df[selected_features[1]]
        
        # Perform Z-test
        z_stat, p_value = ztest(data1, data2)
        
        # Display Z-test results
        st.write(f"Testing the means of `{selected_features[0]}` and `{selected_features[1]}`:")
        st.write(f"Z-statistic: {z_stat:.2f}")
        st.write(f"P-value: {p_value:.4f}")
        
        # Hypothesis conclusion
        alpha = 0.05  # Significance level
        if p_value < alpha:
            st.write("Conclusion: Reject the null hypothesis. The means of the two samples are significantly different.")
        else:
            st.write("Conclusion: Fail to reject the null hypothesis. No significant difference between the means of the two samples.")
    
    # If three or more features are selected, perform ANOVA
    elif len(selected_features) >= 3:
        st.subheader("ANOVA Test")
        
        # Extract data for the selected features
        data_groups = [df[feature] for feature in selected_features]
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*data_groups)
        
        # Display ANOVA results
        st.write(f"Testing the means of the selected features: {', '.join(selected_features)}")
        st.write(f"F-statistic: {f_stat:.2f}")
        st.write(f"P-value: {p_value:.4f}")
        
        # Hypothesis conclusion
        alpha = 0.05  # Significance level
        if p_value < alpha:
            st.write("Conclusion: Reject the null hypothesis. There is a significant difference in means across the selected features.")
        else:
            st.write("Conclusion: Fail to reject the null hypothesis. No significant difference in means across the selected features.")
    
    else:
        st.write("Please select at least two features for hypothesis testing.")

elif visualization == "Normality Testing":
    st.header("Normality Testing")
    
    # Feature Selection for Normality Testing
    selected_features = st.multiselect(
        "Select features for Normality Testing", df.columns.tolist(),
        default=["Sleep Duration"]
    )
    
    # For each selected feature, perform Normality Testing
    for feature in selected_features:
        st.subheader(f"Normality Testing for `{feature}`")
        
        # Extract the data for the feature
        data = df[feature]
        
        # Perform Kolmogorov-Smirnov (KS) Test for normality
        ks_stat, ks_p_value = kstest(data, 'norm', args=(data.mean(), data.std()))
        
        # Perform Shapiro-Wilk Test for normality
        shapiro_stat, shapiro_p_value = shapiro(data)
        
        # Display Results for KS Test
        st.write(f"Kolmogorov-Smirnov Test results:")
        st.write(f"KS-statistic: {ks_stat:.4f}")
        st.write(f"KS-p-value: {ks_p_value:.4f}")
        if ks_p_value < 0.05:
            st.write("Conclusion: Reject the null hypothesis. Data is not normally distributed according to the KS test.")
        else:
            st.write("Conclusion: Fail to reject the null hypothesis. Data appears normally distributed according to the KS test.")
        
        # Display Results for Shapiro-Wilk Test
        st.write(f"Shapiro-Wilk Test results:")
        st.write(f"Shapiro-statistic: {shapiro_stat:.4f}")
        st.write(f"Shapiro-p-value: {shapiro_p_value:.4f}")
        if shapiro_p_value < 0.05:
            st.write("Conclusion: Reject the null hypothesis. Data is not normally distributed according to the Shapiro-Wilk test.")
        else:
            st.write("Conclusion: Fail to reject the null hypothesis. Data appears normally distributed according to the Shapiro-Wilk test.")
    
    # If no features selected
    if len(selected_features) == 0:
        st.write("Please select at least one feature to perform normality testing.")