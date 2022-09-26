# import required libraries
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# beautify the streamlit app
st.title("Penguins Classifier: Supervised ML App")
st.write("This app uses 6 inputs to predict the species of penguin using"
         "a model built on Palmer's Penguin's Dataset. Use the from below"
         "to get started!")


# Ensure that authorized is using this app
password = st.text_input("Enter your Password:")
if password != "Balav":
    st.write("Incorrect Password!")
    st.stop()

    
# set options for getting data inputs from user
penguins_file = st.file_uploader("Upload your own Penguin's Data:")

# if no data is uploaded for training, upload pre-trained model by default
if penguins_file is None:
    # load the pre-trained model as file
    rf_pickle = open("random_forest_penguin.pickle", "rb")
    map_pickle = open("output_penguin.pickle", "rb")
    
    # load the pre-trained model from file
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    st.write("Pre-trained Model: ", rfc)
    rf_pickle.close()
    map_pickle.close()
    
# otherwise take the user's data, clean it, and train a model based on it
else:
    penguins_df = pd.read_csv(penguins_file)
    
    # drop all the missing values
    penguins_df = penguins_df.dropna()
    output = penguins_df['species']
    features = penguins_df[['island', 'bill_length_mm', 'bill_depth_mm',

                           'flipper_length_mm', 'body_mass_g', 'sex']] # features variables
    
    # convert categorical to numerical variables
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    
    # Split the dataset into training and test set
    X_train,x_test, y_train,y_test = train_test_split(features, output, test_size=0.8)
    
    # instantiate RandomForestClassifier model
    rfc = RandomForestClassifier(random_state=15)
    
    # Fit & train, predict and calculate the accuracy of the trained model
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    
    # display ouput on the streamlit app
    st.write("We trained a Random Forest Model on these data "
             "it has a score of **{}**! Use the "
             "inputs below to try out the model.".format(score))
    
    
# Create a form where user can submit their inputs values for model
# The model trains on inputs data and makes a predictions
with st.form("User Inputs"):
    island = st.selectbox("Penguins Island:", options=['Biscoe', 'Dream', 'Torgerson'])
    
    sex = st.selectbox("Sex:", options=["Female", "Male"])
    
    bill_length = st.number_input("Bill length (mm):", min_value=0)
    
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)

    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)

    body_mass = st.number_input('Body Mass (g)', min_value=0)
    
    st.form_submit_button() # submit the inputs for model
    
    # turn "Island" column from categorical to numerical variable
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == "Biscoe":
        island_biscoe = 1
    elif island == "Dream":
        island_dream = 1
    elif island == "Torgenson":
        island_torgenson = 1
        
    # turn "Sex" column from categorical to numerical variable
    sex_female, sex_male = 0, 0
    if sex == "Female":
        sex_female = 1
    elif sex == "Male":
        sex_male = 1
        
# Make a predictions and display it on the Streamlit App
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,

                               body_mass, island_biscoe, island_dream,

                               island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]

st.subheader("Predicting Your Penguin's Species:")

st.write("The predicted Penguin is of the **{}** species".format(prediction_species))

st.write("Prediction was done using **Random Forest Classifer** Model. "
         "The importance of features in the Model is shown as below:")
st.image('feature_importance.png')

st.markdown("**bill length, bill depth, body mass** and **flipper length** are the most important variables according to our random forest model.") 

st.write("* * * * * * ** * * * * * * * * * * * * * * * * * * *")


# Display some for graphs of important features in user uploads a file
if penguins_file is not None:
    st.write('Below are the histograms for each continuous variable '

            'separated by penguin species. The vertical line '

            'represents your the inputted value.')

    # Create a 1st distibution chart
    fig1, ax1 = plt.subplots()

    ax1 = sns.displot(x=penguins_df['bill_length_mm'], hue=penguins_df['species'])

    plt.axvline(bill_length)
    plt.title("Bill length by Species")

    st.pyplot(ax1)

    # Create a 2nd distribution chart
    fig2, ax2 = plt.subplots()

    ax2 = sns.displot(x=penguins_df['bill_depth_mm'], hue=penguins_df['species'])

    plt.axvline(bill_depth)
    plt.title("Bill Depth by Species")

    st.pyplot(ax2)

    # create a 3rd distribution chart
    fig3, ax3 = plt.subplots()

    ax3 = sns.displot(x=penguins_df['flipper_length_mm'], hue=penguins_df['species'])

    plt.axvline(flipper_length)
    plt.title("Flipper Length by Species")

    st.pyplot(ax3)
