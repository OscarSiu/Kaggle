"""""""""""""""""""""""""""
# Binary Classification

Accuracy = number_correct / total [0-1]
Cross-entropy = distance between probabilities
"""""""""""""""""""""""""""
import pandas as pd
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers


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

""""""""""""""""""""""""""""""""""""""""""""""""""""
Dropout prevents overfitting
rate defines percentage of the input units to shut off

Batch Normalization helps correct training that is slow or unstable
"""""""""""""""""""""""""""""""""""""""""""""""""""""

model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    #layers.Dropout(rate = 0.3),
    layers.BatchNormalization(),
    layers.Dense(16, activation='relu'),
    #layers.Dropout(0.3),
    layers.BatchNormalization(),
    
    layers.Dense(1),
    #layers.Dense(1, activation='sigmoid'),
])

model.compile(
   # optimizer='adam',
   # loss='binary_crossentropy',
   # metrics=['binary_accuracy'],
    optimizer='sgd',
    loss='mae',
    metrics=['mae'],

)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=50,
    callbacks=[early_stopping],
    #verbose=0, # hide the output because we have so many epochs
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[:, ['loss', 'val_loss']].plot()
#history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

"""
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))
"""

print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
