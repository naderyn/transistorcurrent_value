import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# Load the dataset
file_path = 'C:\Users\EGY 10\Downloads\simulated_data_nmos (1).csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns and preprocess
features = ['vds', 'L(um)', 'W(um)', 'drain_length(um)', 'temperature', 'vgs', 'vsb', 'corner']
target = 'id(uA)'
data = data[features + [target]]
data = pd.get_dummies(data, columns=['corner'], drop_first=True)

# Split and normalize
X = data.drop(target, axis=1)
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(256, activation='relu', kernel_regularizer='l2', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[reduce_lr, early_stop])

# Evaluate
loss = model.evaluate(X_test, y_test)
print(f'MAPE on Test Data: {loss}%')
