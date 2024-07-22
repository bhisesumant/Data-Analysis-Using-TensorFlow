import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

data = pd.read_csv('cars.csv')

data

data.info()

print("Total missing values:", data.isna().sum().sum())
print("Columns with missing values:", data.columns[data.isna().sum() > 0].values)
data['engine_capacity'].dtype
data['engine_capacity'] = data['engine_capacity'].fillna(data['engine_capacity'].mean())

print("Total missing values:", data.isna().sum().sum())

for column in data.columns:
    if data.dtypes[column] == 'bool':
        data[column] = data[column].astype(np.int)

data = data.drop('model_name', axis=1)
data = data.drop('engine_fuel', axis=1)


def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

onehot_columns = [
    'manufacturer_name',
    'transmission',
    'color',
    'body_type',
    'state',
    'location_region'
]

onehot_prefixes = [
    'm',
    't',
    'c',
    'b',
    's',
    'l'
]

data = onehot_encode(
    data,
    columns=onehot_columns,
    prefixes=onehot_prefixes
)

print("Remaining non-numeric columns:", (data.dtypes == 'object').sum())


data['engine_type'].unique()

engine_mapping = {
    'gasoline': 0,
    'diesel': 1,
    'electric': 2
}

data['engine_type'] = data['engine_type'].replace(engine_mapping)

data['drivetrain'].unique()

drivetrain_mapping = {'all': 0, 'front': 1, 'rear': 2}

data['drivetrain'] = data['drivetrain'].replace(drivetrain_mapping)


y = data['drivetrain'].copy()
X = data.drop('drivetrain', axis=1).copy()

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

X.shape

inputs = tf.keras.Input(shape=(111,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dense(8, activation='relu')(x)
x = tf.keras.layers.Dense(4, activation='relu')(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.4,
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

model.evaluate(X_test, y_test)