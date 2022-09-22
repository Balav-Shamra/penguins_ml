"""
Generate a feature Importance graph once and save it to the folder.

Do not include this program inside Streamlit Web app, since it will have to regenerate every it reloads.

Hence, run it separetely once to generate a figure and save it.
"""

# Requred libraries
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
import streamlit as st
import pickle # to save the trained model


# import and clean the Penguins dataset
penguin_df = pd.read_csv('../Dataset/penguins.csv') 
penguin_df.dropna(inplace=True) 
output = penguin_df['species'] 
features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']] 
features = pd.get_dummies(features)  # convert categorical to numerical features
output, uniques = pd.factorize(output) 

x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8) 
rfc = RandomForestClassifier(random_state=15) 
rfc.fit(x_train, y_train) 
y_pred = rfc.predict(x_test) 
score = accuracy_score(y_pred, y_test) 
print('Accuracy Score of the Model: {}'.format(score)) 

# Save the trained model for future useage
rf_pickle = open('random_forest_penguin.pickle', 'wb') 
pickle.dump(rfc, rf_pickle) 
rf_pickle.close() 

output_pickle = open('output_penguin.pickle', 'wb') 
pickle.dump(uniques, output_pickle) 
output_pickle.close() 


# plot a feature importance graph and save it to import in the Streamlit Web App
fig, ax = plt.subplots() 

ax = sns.barplot(x=rfc.feature_importances_, y=features.columns) 
plt.title('Feature Imporance in Penguin Species predictions') 
plt.xlabel('Importance') 
plt.ylabel('Features') 
plt.tight_layout() 

# save the generated plot to import it in the streamlit Web app later
fig.savefig('feature_importance.png')
