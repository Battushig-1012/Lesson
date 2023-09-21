#Opportunity evaluation: easy to develop and change

#CRISP-DM Business Understanding: it can attachment use.

#Validation plan: every one can use it

#ML system design: streamlit can handle large working process

#Potential risks in production: only works with csv

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('CCPP_data.csv')

df = load_data()

st.write(df)

# Split the data into features (X) and target variable (y)
X = df.drop("PE", axis=1)
y = df["PE"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Perform PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Create a DataFrame for the PCA results
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['PD'] = y

# Streamlit app
st.title("Credit Risk Prediction App")
st.write("This app predicts credit risk using a Naive Bayes model.")

# Display the dataset
st.subheader("Loan Data")
st.write(df_pca)

# Display the scatter plot
st.subheader("Scatter Plot (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='PD', palette='viridis', ax=ax)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Scatter Plot (PCA) - Credit Risk Prediction')
st.pyplot(fig)

# Display the classification report
st.subheader("Classification Report")
predictions = nb_model.predict(X_test)
report = classification_report(y_test, predictions)
st.write(report)

# Display the confusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)
st.write(cm)

# Display the accuracy
st.subheader("Accuracy")
accuracy = accuracy_score(y_test, predictions)
st.write(accuracy)
