import pandas as pd
import numpy as np
import streamlit as st
import tabula
from PyPDF2 import PdfReader
import io
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
import plotly.express as px

from streamlit.components.v1 import html
from ipyvizzu import Chart, Data, Config, Style, DisplayTarget
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statistics
import re
import toml
from datetime import datetime


st.set_page_config(
   page_title="Khan Statements",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.title("Зээлийн тооцоолол")

def khan_bank(uploaded_file):
    with st.spinner('Processing...'):
        def count_pdf_pages(file): 
            pdf_reader = PdfReader(file)
            total_pages = len(pdf_reader.pages)
            return total_pages

        # Convert UploadedFile object to a file-like object
        pdf_file = io.BytesIO(uploaded_file.read())

        total_pages = int(count_pdf_pages(pdf_file)) + 1

        def extract_page_1_data():
            page_1 = tabula.read_pdf(pdf_file, pages=1,
                                     relative_area=True,
                                     relative_columns=True,
                                     area=[10, 0, 95, 100],
                                    columns=[14, 22, 38, 49, 60, 73, 88])
            df1 = pd.concat([pd.DataFrame(table) for table in page_1], ignore_index=True)
            df1.columns = ['Тайлант огноо',
        'Салбар', 
        'Эхний үлдэгдэл', 
        'Зарлага', 
        'Орлого', 
        'Эцсийн үлдэгдэл', 
        'Гүйлгээний утга', 
        'Харьцсан данс']

            return df1

        def extract_other_pages_data():
            total_df2 = pd.DataFrame(columns=['Тайлант огноо',
        'Салбар', 
        'Эхний үлдэгдэл', 
        'Зарлага', 
        'Орлого', 
        'Эцсийн үлдэгдэл', 
        'Гүйлгээний утга', 
        'Харьцсан данс'])
            for page_number in range(2, total_pages): 
                page_other = tabula.read_pdf(pdf_file, pages=page_number,
                                            relative_area=True,
                                            relative_columns=True,
                                            area=[3, 0, 95, 100],
                                            columns=[14, 22, 38, 49, 60, 73, 88])
                if page_other:  # Check if tables are present on the page
                    df2 = pd.concat([pd.DataFrame(table) for table in page_other], ignore_index=True)
                    df2.columns = ['Тайлант огноо',
        'Салбар', 
        'Эхний үлдэгдэл', 
        'Зарлага', 
        'Орлого', 
        'Эцсийн үлдэгдэл', 
        'Гүйлгээний утга', 
        'Харьцсан данс']
                    total_df2 = pd.concat([total_df2, df2], ignore_index=True)
            return total_df2


        p1 = extract_page_1_data()
        p2 = extract_other_pages_data()
        p1 = p1.copy()
        p2 = p2.copy()

        total_df = pd.concat([p1, p2], ignore_index=True)

        total_df =total_df[['Тайлант огноо',
        'Салбар', 
        'Эхний үлдэгдэл', 
        'Зарлага', 
        'Орлого', 
        'Эцсийн үлдэгдэл', 
        'Гүйлгээний утга', 
        'Харьцсан данс']]
        
        return total_df
    


def df_clearing(result_df):
    result_df['Тайлант огноо'].fillna(method='ffill', inplace=True)
    result_df['Тайлант огноо'] = pd.to_datetime(result_df['Тайлант огноо'])
    result_df['Тайлант огноо'] = result_df['Тайлант огноо'].dt.strftime('%Y/%m')
    
    result_df['Зарлага'] = result_df['Зарлага'].astype(str).str.replace(',', '').astype(float)
    result_df['Зарлага'].fillna(0, inplace=True)
    result_df['Зарлага'] = result_df['Зарлага'] * -1

    result_df['Орлого'] = result_df['Орлого'].astype(str).str.replace(',', '').astype(float)
    result_df['Орлого'].fillna(0, inplace=True)

    result_df['Гүйлгээний утга'].fillna(value='No_data', inplace=True)
    
    return result_df

def information(uploaded_files):
    page_1 = tabula.read_pdf(uploaded_files, pages=1,
                                relative_area=True,
                                relative_columns=True,
                                area=[2, 0, 95, 100],
                                columns=[12, 19, 49, 60, 70, 87, 100])
    df_info = pd.concat([pd.DataFrame(table) for table in page_1], ignore_index=True)
    df_info.columns = ['Тайлант огноо', 
                       'Салбар', 
                       'Зарлага',
                        'Орлого', 
                        'Эцсийн үлдэгдэл', 
                        'Гүйлгээний утга',
                        'Харьцсан данс']
    name = df_info.iloc[:2, 2].tolist()
    return name


uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    
    final_df = pd.DataFrame()  # Create an empty DataFrame to store the combined data

    for uploaded_file in uploaded_files:
        result_df = khan_bank(uploaded_file)
        final_df = pd.concat([final_df, result_df], ignore_index=True)
        df_info1 = information(uploaded_file)



df = (df_clearing(final_df))

df = df.sort_values(by='Тайлант огноо')

df.copy()

st.subheader('', divider='rainbow')

st.write(df_info1)

def duplicate():
    # Calculate the number of credit transactions for each dance style
    sums_by_duplicate_count = df.groupby('Харьцсан данс')['Орлого'].count()

    # Calculate the total amount of credit transactions for each dance style
    sums_by_duplicate_sum = df.groupby('Харьцсан данс')['Орлого'].sum()

    # Combine the two DataFrames into a single DataFrame
    combined_df = pd.concat([sums_by_duplicate_count, sums_by_duplicate_sum], axis=1)

    # Rename the columns
    combined_df.columns = ['Count', 'Sum']

    # Sort the DataFrame by the 'Sum' column in descending order
    combined_df = combined_df.sort_values(by='Sum', ascending=False)

    return combined_df

# Display the results using an expander
with st.expander("Давхардсан гүйлгээнүүдийг харах"):
    st.write(duplicate())

find_names = ['zeel', 'зээл', 'bet']

filtered_sum = df[df['Гүйлгээний утга'].isin(find_names)]['Орлого'].count()
st.warning(find_names)
st.write(filtered_sum)

#---------------------------------------------------------------------------- Эндээс доош дата clearning 

st.subheader('', divider='rainbow')



col1, col2, col3 = st.columns(3)


fig = px.bar(df, x='Тайлант огноо', y='Зарлага')
with col1:
    st.warning("Тухайн сард нийт Зарлага")
    st.plotly_chart(fig)


fig = px.bar(df, x='Тайлант огноо', y='Орлого')
with col2:
    st.warning("Өдөр бүрийн үлдэгдэл")
    st.plotly_chart(fig)


# Assuming df['In_Out'] contains numerical data (if not, convert it to numerical or handle missing values)
X = df[['Орлого']]

# Perform k-means clustering with, for example, 3 clusters
num_clusters = st.slider("Хэдэн кластерт хуваах эсэхийг сонгоно уу:", min_value=2, max_value=6, value=3, format="%d")
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(X)

# Add cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

df = df.copy()

with st.expander("Data frame"):
    st.write(df)

# Create a scatter plot using Plotly Express
fig = px.scatter(df, x='Тайлант огноо', y='Орлого', color='Cluster', size='Орлого')
st.subheader('', divider='rainbow')
# Display the plot using Streamlit
with col3:
    st.warning("Cluster-н хувьд орлогын хэмжээг харвал")
    st.plotly_chart(fig)


income_sum = df.groupby('Тайлант огноо')['Орлого'].sum()

Cost_sum = df.groupby('Тайлант огноо')['Зарлага'].sum()

Minus = income_sum - Cost_sum


col1, col2, col3 = st.columns(3)

with col1:
    st.title('Орлого')
    st.warning("Сар бүрийн орлого")
    st.write(income_sum)

with col2:
    st.title('Зарлага')
    st.warning("Сар бүрийн зарлага")
    st.write(Cost_sum)

with col3:
    st.title('Зөрүү')
    st.warning("Тухайн сар бүрийн зөрүү")
    st.write(Minus)

st.subheader('', divider='rainbow')

col1, col2 = st.columns(2)

with col1:
    filtered_df = df[df['Орлого'] != 0]
    fig = px.histogram(filtered_df, x='Орлого', nbins=10, title='Distribution of inflow')
    st.plotly_chart(fig)

with col2:
    filtered_df = df[df['Зарлага'] != 0]
    fig = px.histogram(filtered_df, x='Зарлага', nbins=10, title='Distribution of outflow')
    st.plotly_chart(fig)


col1, col2 = st.columns(2)



df['Тайлант огноо'] = pd.to_datetime(df['Тайлант огноо'])
df['month'] = df['Тайлант огноо'].dt.month
number_of_months = df['month'].nunique()

# Calculate the average income for each cluster
df['Average'] = df['Орлого'] / number_of_months

# Update the cluster summary table
cluster_summary = df.groupby('Cluster')['Average'].sum().reset_index()
cluster_summary_f = cluster_summary.applymap('{:,.0f}'.format)

st.subheader('', divider='rainbow')

st.warning("Сард хуваасан дундаж бөгөөд тохирох утгаа сонгох")

st.table(cluster_summary_f)

number = st.number_input('Insert a number')

save_button = st.button("Хадгалах")


