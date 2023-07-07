import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Path of the file to read
iowa_file_path = 'ABC.csv'
home_data = pd.read_csv(iowa_file_path)   

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Specify Prediction Target
y = home_data.SalePrice

#Input features
# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review data
# print summary statistics from X
#print(X.describe())
# print the top few lines
#print(X.head())

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Decision Tree

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Build, Specify and fit Model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(train_X, train_y)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Make predictions with validation data
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Deal with underfitting & overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
for i in candidate_max_leaf_nodes:
    my_mae = get_mae(i, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(i, my_mae))

'''
Mean Absolute Error (MAE) = actual - predicted
'''

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = 100

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
iowa_model.fit(train_X, train_y)

#Make predictions with validation data
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

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
