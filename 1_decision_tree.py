# Kaggle Intro to Machine Learning

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Path of the file to read
file_path = 'Football_teams.csv'
football_data = pd.read_csv(file_path)   

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Specify Prediction Target
y = football_data.Rating

# Create the list of features below
feature_names = ["Goals","Shots pg","Possession%","Pass%","AerialsWon"]

# Select data corresponding to features in feature_names
X = football_data[feature_names]

# Review data
# print summary statistics from X
print(X.describe())
# print the top few lines
print(X.head())

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Decision Tree

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Build, Specify and fit Model
model = DecisionTreeRegressor(random_state=1)

# Fit the model
model.fit(train_X, train_y)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Make predictions with validation data
val_predictions = model.predict(val_X)

# print the top few validation predictions
print(val_predictions[0:5])
# print the top few actual rating from validation data
print(y.head())

val_mae = mean_absolute_error(val_predictions, val_y)
print("\nValidation MAE: {:,.3f}\n".format(val_mae))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Deal with underfitting & overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 10, 20, 25, 50, 75, 100]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for i in candidate_max_leaf_nodes:
    my_mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %.3f" %(i, my_mae))

'''
Mean Absolute Error (MAE) = actual - predicted
'''

# Store the best value of max_leaf_node
best_tree_size = 5

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Using best value for max_leaf_nodes
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)

#Make predictions with validation data
val_predictions = final_model.predict(X)
val_mae = mean_absolute_error(val_predictions, y)
print("Validation MAE for best value of max_leaf_nodes: {:,.3f}".format(val_mae))

#Validation
#print(iowa_model.predict(X.head()))
#print(y.head().tolist())

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
