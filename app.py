import streamlit as st
import pickle
import numpy as np
import pandas as pd  

# import the model
laptop_model = pickle.load(open('laptop_predicter_model.pkl','rb'))
df = pickle.load(open('laptop.pkl','rb'))

st.title('Laptop Price Predictor')

# brand...
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop...
type = st.selectbox('Type', df['TypeName'].unique())

# ram
ram = st.selectbox('Ram(in GB)', df['Ram'].unique())

# weight
weight = st.number_input('Weight of the laptop')

# touchscreen...
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS Panel', ['No', 'Yes'])

# screen size....
screen_size = st.number_input('Screen Size')

# resolutions
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

HDD = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

SSD = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

GPU = st.selectbox('GPU', df['Gpu'].unique())

os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):
    # query...
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    ppi = (((x_res**2) + (y_res**2))**0.5) / screen_size

    # Create a query array with the selected inputs
    query = np.array([company, type, ram, GPU, weight, touchscreen, ips, ppi, cpu, SSD, HDD, os])

    # Convert the query into a pandas DataFrame with column names
    # Ensure these column names match the ones used for training


    columns = ['Company', 'TypeName', 'Ram', 'Gpu','Weight', 'touchscreen', 'IPS', 'PPI', 'Cpu Brand','SSD', 'HDD','OS']
    query_df = pd.DataFrame([query], columns=columns)

    # Use the model to predict the price
    predicted_price = laptop_model.predict(query_df)

    # Display the predicted price (Exponentiate the prediction to get the actual price)
    st.header(f'Predicted Laptop Price: ₹{int(np.exp(predicted_price[0]))}')
