# Required libraries
import streamlit as st
import pandas as pd
import pickle

st.title("Utilizing a Pre-Trained ML model in Streamlit App")

st.header("Penguins Species Classifier with Pre-Trained Models")

st.write("This app uses 6 inputs to predict the species of penguin using"

         "a model built on the **Palmer's Penguin's dataset**. Use the form below"

         " to get started!")

# load the pretrained RandomForestClassifier Model
# read in the saved files contained trained models
rf_pickle = open("random_forest_penguin.pickle", "rb") # rb -> read bytes
map_pickle = open("output_penguin.pickle", "rb")

# close the file
# rf_pickle.close()
# map_pickle.close()

# load the model from the read files
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
# convert to dataframe to give meaningful column name
unique_penguin_mapping_labels = pd.DataFrame(unique_penguin_mapping)
unique_penguin_mapping_labels.columns = ["Penguin's species"]

# Display the loaded model to the Streamlit App
st.write(rfc)
st.write("Target mapping for each ouput label is as below:")
st.write(unique_penguin_mapping_labels)

# Taking the user inputs as values for different Featues
island = st.selectbox("Penguin Island:",options=['Biscose', 'Dream', 'Torgerson'])
sex = st.selectbox("Select Gender:", options=['Female',"Male"])
bill_length = st.number_input("Bill Length (mm):", min_value=0)
bill_depth = st.number_input("Bill Depth (mm):", min_value=0)
flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
body_mass = st.number_input('Body Mass (g)', min_value=0)

# Display the user inputs
st.write("User Inputs are: {}".format([island,sex,bill_length, bill_depth,flipper_length,body_mass]))

# Convert the "island" & "sex" column into to correct format as you did in traing set to fit to the model for predicton
# 1. Converting "island" column
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1
    
# 2. Converting "sex" column
sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# Fitting our new data to the trained model to make predictions
new_prediction = rfc.predict(
    [[bill_length, bill_depth, flipper_length, body_mass, 
      island_biscoe, island_dream, island_torgerson, sex_female, sex_male]]
    )

# Create a horizontal line
st.write("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
st.write("~~> Predicted label: **{}**".format(new_prediction[0]))

prediction_species = unique_penguin_mapping[new_prediction][0]

st.write(" ~~> Predicted Penguin is of the **{}** Species".format(prediction_species))
st.write("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")