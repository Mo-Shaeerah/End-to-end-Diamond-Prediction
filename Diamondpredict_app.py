# Import libraries
import streamlit as st
from PIL import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import plotly.express as px

import pickle
import xgboost
import catboost

# Step 1: Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Step 2: Set the title
st.title("Diamond Price Prediction App 💎")

# Step 3: Show diamond image
image = Image.open("Images/DiamondImage.jpg")
st.image(image, caption="Diamond", use_column_width=True)

# Step 4: Load and display a sample of data
data = pd.read_csv("Data/Diamond-train.csv", index_col="Id")
st.subheader("Sample Data")
st.dataframe(data.sample(10))

# Step 5: Visualizations
fig, axes = plt.subplots(1, 4, figsize=(25, 10))

fig = px.box(data.price, orientation="h", template="plotly_dark")
st.plotly_chart(fig)

sns.boxplot(x= data.price, showmeans=True, color='red', ax=axes[0])
sns.set_style('dark')
sns.violinplot(x= data.price, ax=axes[1])
sns.histplot(x= data.price, bins=20, color="red", kde=True, ax=axes[2])
sns.kdeplot(x= data.price, fill=True, color='red', ax=axes[3]);

st.pyplot()

selected_attribute = st.selectbox("Select Attribute to Visualize", ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"])

fig, axes = plt.subplots(1, 4, figsize=(25, 10))

if selected_attribute == "carat":
    # Visualizations for carat attribute
    sns.boxplot(x=data.carat, showmeans=True, color='red', ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x=data.carat, ax=axes[1])
    sns.histplot(x=data.carat, bins=20, color="blue", kde=True, ax=axes[2])
    sns.kdeplot(x=data.carat, fill=True, color='red', ax=axes[3])
elif selected_attribute == "cut":
    #Visualization for cut attribute
    sns.barplot(x =data.cut, y =data.price, ax =axes[0])
    sns.set_style('dark')
    axes[1].pie(x =data.cut.value_counts(), labels =data.cut.unique(), autopct='%1.1f%%', shadow=True)
    sns.violinplot(x= data.cut, y= data.price, ax =axes[2])
    sns.stripplot(data= data, x='cut', y='price', color="red", ax =axes[3])
elif selected_attribute == "color":
    sns.barplot(x =data.color, y =data.price, ax =axes[0])
    sns.set_style('dark')
    axes[1].pie(x =data.color.value_counts(), labels =data.color.unique(), autopct='%1.1f%%', shadow=True)
    sns.violinplot(x= data.color, y= data.price, ax =axes[2])
    sns.stripplot(data= data, x='color', y='price', color="red", ax =axes[3])
elif selected_attribute == "clarity":
    sns.barplot(x =data.clarity, y =data.price, ax =axes[0])
    sns.set_style('dark')
    axes[1].pie(x =data.clarity.value_counts(), labels =data.clarity.unique(), autopct='%1.1f%%', shadow=True)
    sns.violinplot(x= data.clarity, y= data.price, ax =axes[2])
    sns.stripplot(data= data, x='clarity', y='price', color="red", ax =axes[3])
elif selected_attribute == "depth":
    # Visualizations for depth attribute
    sns.regplot(x= data.price, y= data.depth, line_kws={"color": "red"}, ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x= data.depth, ax=axes[1])
    sns.histplot(x= data.depth, bins=20, color="red", kde=True, ax=axes[2])
    sns.kdeplot(x= data.depth, fill=True, color='red', ax=axes[3])
elif selected_attribute == "table":
    sns.regplot(x= data.price, y= data.table, line_kws={"color": "red"}, ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x= data.table, ax=axes[1])
    sns.histplot(x= data.table, bins=20, color="red", kde=True, ax=axes[2])
    sns.kdeplot(x= data.table, fill=True, color='red', ax=axes[3])
elif selected_attribute == "x":
    sns.regplot(x= data.price, y= data.x, line_kws={"color": "red"}, ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x= data.x, ax=axes[1])
    sns.histplot(x= data.x, bins=20, color="red", kde=True, ax=axes[2])
    sns.kdeplot(x= data.x, fill=True, color='red', ax=axes[3])
elif selected_attribute == "y":
    sns.regplot(x= data.price, y= data.y, line_kws={"color": "red"}, ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x= data.y, ax=axes[1])
    sns.histplot(x= data.y, bins=20, color="red", kde=True, ax=axes[2])
    sns.kdeplot(x= data.y, fill=True, color='red', ax=axes[3])
elif selected_attribute == "z":
    sns.regplot(x= data.price, y= data.z, line_kws={"color": "red"}, ax=axes[0])
    sns.set_style('dark')
    sns.violinplot(x= data.z, ax=axes[1])
    sns.histplot(x= data.z, bins=20, color="red", kde=True, ax=axes[2])
    sns.kdeplot(x= data.z, fill=True, color='red', ax=axes[3])
st.pyplot()

# Step 6: Add "Done by Mohammed"
st.markdown("Project Link: [project url](https://github.com/Mo-Sa-Shaeerah/End-to-end-Diamond-Prediction)")
print()
st.markdown("Done by: [Mohammed Salf Shaeerah](https://github.com/Mo-Sa-Shaeerah)")

# Step 7: Characteristics of Diamond in Sidebar
# Set sidebar title
st.sidebar.title("Charact.. of Diamond 💎")

# Divide the sidebar into two columns
col1, col2 = st.sidebar.columns(2)  

# First column in the sidebar
with col1:
    carat = st.number_input('Carat Weight:', min_value=0.5, max_value=10.0, value=1.0)
    cut = st.selectbox('Cut Rating:', ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'])
    color = st.selectbox('Color Rating:', ['D', 'E', 'F', 'G',  'H', 'I', 'J'])
    clarity = st.selectbox('Clarity Rating:', ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])
    depth = st.number_input('Diamond Depth %:', min_value=0.5, max_value=100.0, value=1.0)

# Second column in the sidebar
with col2:
    table = st.number_input('Diamond Table %:', min_value=0.5, max_value=100.0, value=1.0)
    x = st.number_input('Diamond Length in mm:', min_value=0.5, max_value=100.0, value=1.0)
    y = st.number_input('Diamond Width in mm:', min_value=0.5, max_value=100.0, value=1.0)
    z = st.number_input('Diamond Height in mm:', min_value=0.5, max_value=100.0, value=1.0)

# Stpe 8: Loading up the model
model = xgboost.XGBRegressor()
model.load_model('Model/xgboost_model.json')

#Caching the model for faster loading
@st.cache_resource

# Step 9: Build a Predict function
def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Encode object variables with numbers
    obj_cut = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
    num_cut = [1, 2, 3, 4, 5]
    cut_mapping = dict(zip(obj_cut, num_cut))
    cut = cut_mapping[cut]
      
    obj_color = ['D', 'E', 'F', 'G', "H", 'I', 'J']
    num_color = [1, 2, 3, 4, 5, 6, 7]
    color_mapping = dict(zip(obj_color, num_color))
    color = color_mapping[color]
        
    obj_clarity = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1']
    num_clarity = [1, 2, 3, 4, 5, 6, 7, 8]
    clarity_mapping = dict(zip(obj_clarity, num_clarity))
    clarity = clarity_mapping[clarity]

    lists = [[carat, cut, color, clarity, depth, table, x, y, z]]

    df = pd.DataFrame(lists, columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])

    # making predictions using the trained model
    prediction = model.predict(df)
    result = int(prediction)

    return result

# Step 10: Prediction Button
if st.sidebar.button("Predict"):
    # Perform the prediction
    prediction = predict(carat, cut, color, clarity, depth, table, x, y, z)

    # Display the predicted price
    st.sidebar.success(f'The Predicted Price Of 💎 is {prediction} USD')