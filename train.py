from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os
import numpy as np


label_map = dict()
label_map['texting']=0
label_map['idle']=1

# print(label_map)

# Path for exported data
DATA_PATH = os.path.join("MP_Data")

# Actions
actions = np.array(['texting', 'idle'])

# Samples
no_sequences = 30

# Frames
sequence_length = 60

# Folder start
start_folder = 0

sequences, labels = [],[]
for action in actions:
	for sequence in range(no_sequences):
		window = []
		for frame_num in range(sequence_length):
			res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
			window.append(res)
		sequences.append(window)
		labels.append(label_map[action])
# print(np.array(sequences).shape)

X = np.array(sequences)

y = to_categorical(labels).astype(int)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05)


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
model.summary()
model.save('pose.h5')