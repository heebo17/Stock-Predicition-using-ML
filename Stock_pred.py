import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# To Do List:


def main():
    #Load Data
    name = "FB"
    company = name

    start = dt.datetime(2012,1,1)
    end = dt.datetime(2021,1,1)

    data = web.DataReader(company, "yahoo", start, end)

    #prepare Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

    prediction_days = 60 #How many days I want to take to predict
   
  

    x_train = []
    y_train =[]
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x,0])


    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    print("x_train", x_train.shape)
    print("x_train.shaoe[0]",x_train.shape[0])
    print("X_train.shape[1]", x_train.shape[1])

    #Build Model
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    #train the model
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    

    #Test accuracy of the model on new data

    test_start = dt.datetime(2021,1,1) #new data which the model hasnt trained
    test_end = dt.datetime.now()

    test_data = web.DataReader( name, "yahoo", test_start, test_end)
    actual_price = test_data["Close"].values
    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape( -1, 1)
    model_inputs = scaler.transform(model_inputs)

    #VALIDATE THE MODEL   
    x_test = []
    print("length model inputs", len(model_inputs))
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    

    x_test = np.array(x_test)

    print("X_test", x_test)
    print("x_test.shape[0]",x_test.shape[0])
    print("X_test.shape[1]", x_test.shape[1])

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print("X_test shape: ", x_test.shape)

    
    predicted_prices = model.predict(x_test)
    print("predicted_prices: ", predicted_prices.shape)
    predicted_prices = np.reshape(predicted_prices, (predicted_prices.shape[0], predicted_prices.shape[1]))
    predicted_prices = scaler.inverse_transform(predicted_prices)


    #PLOT THE TEST PREDICTIONS
    plt.plot(actual_price, color="black", label="Actual {company} Price")
    plt.plot(predicted_prices, color="green", label="predicted {company} prices")
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend()
    plt.show()



    #Predict new Values
    real_data = ([model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1)]], 0)
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")

    return 0

main()