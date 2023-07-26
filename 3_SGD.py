# The Optimizer - Stochastic Gradient Descent
# Solving regression problems

import pandas as pd
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers, callbacks

football_team = pd.read_csv('../input/football-teams-rankings-stats/Football teams.csv', usecols = ["Goals","Shots pg","Possession%","Pass%","AerialsWon", "Rating"])

# Create training and validation splits
df_train = football_team.sample(frac=0.7, random_state=0)
df_valid = football_team.drop(df_train.index)
display(df_train.head(5))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('Rating', axis=1)
X_valid = df_valid.drop('Rating', axis=1)
y_train = df_train['Rating']
y_valid = df_valid['Rating']

print(X_train.shape)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Early stopping to deal with overfitting and underfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.1, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# train the model 
model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=[5]),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1),
])


#Compile Optimizer and loss function
# loss function = [MAE, MSE, Huber loss]
model.compile(
    optimizer='adam', 
    loss='mae',
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=16,
    epochs=10,
)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
