from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Define paths for the saved model
MODEL_FILEPATH = "path/to/your/saved_model.h5"

def home(request):
    return render(request, "Home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    # Load the housing dataset
    data = pd.read_csv("C:/Users/lahar/Downloads/archive (4)/USA_Housing.csv")
    data = data.drop(['Address'], axis=1)  # Drop the 'Address' column
    X = data.drop('Price', axis=1)
    Y = data['Price']

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Check if the model exists
    if not os.path.exists(MODEL_FILEPATH):
        # Build the neural network model
        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Add early stopping and model checkpointing to save the best model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(MODEL_FILEPATH, save_best_only=True, monitor='val_loss', mode='min')

        # Train the model
        model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=32,
                  callbacks=[early_stopping, checkpoint], verbose=1)
    else:
        # Load the pre-trained model
        model = load_model(MODEL_FILEPATH)

    # Prediction logic
    def predict_price(request):
        try:
            # Fetch input parameters from the GET request
            var1 = float(request.GET.get('n1'))
            var2 = float(request.GET.get('n2'))
            var3 = float(request.GET.get('n3'))
            var4 = float(request.GET.get('n4'))
            var5 = float(request.GET.get('n5'))

            # Prepare input data for prediction
            input_data = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)
            input_data = scaler.transform(input_data)  # Normalize input data

            # Make the prediction
            pred = model.predict(input_data)
            pred = round(pred[0][0])  # Round the prediction to a whole number

            # Prepare the response
            price = f"The predicted price is ${pred}"
            return HttpResponse(price)

        except Exception as e:
            # Handle any exceptions
            return HttpResponse(f"An error occurred: {str(e)}", status=400)

    return predict_price(request)
