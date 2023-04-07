from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Reshape, AveragePooling2D , AveragePooling3D, MaxPooling3D, BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import ConvLSTM2D

from keras.layers import Conv2D, Dense, TimeDistributed, LSTM, Flatten, Input

def ConvLSTM2D_model(n_outputs):
    nFilters = 32
    model = Sequential()
    model.add(ConvLSTM2D(filters = nFilters, kernel_size=(11,11), padding="same",return_sequences=True, activation="relu"))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters = nFilters, kernel_size=(5,5), padding="same",return_sequences=True, activation="relu"))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters = nFilters, kernel_size=(3,3), padding="same",return_sequences=True, activation="relu"))
    model.add(BatchNormalization())
    model.add(AveragePooling3D((1,28,28), padding='same'))

    model.add(Reshape((-1, nFilters)))
    model.add(TimeDistributed(Dense(n_outputs,activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
