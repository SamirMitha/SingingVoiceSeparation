import glob, os
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pickle
from model import SVSRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, EarlyStopping
from data_generator import DataGenerator
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Default
n_channels = 1
batch_size = 1024
epochs = 30
learning_rate = 1e-4
model_loss = 'mean_squared_error'
monitor = 'val_loss'
train_split = 0.5
validation_split = 0
test_split = 0.5
checkpoint = 00

# Directories
train_path = '/media/samir/Secondary/Voice Separation/RNN/training/mix/'
model_path = '/media/samir/Secondary/Voice Separation/RNN/models'
model_name = 'SVSRNN_Default_DSD100A'

# Create output directories
if(not os.path.isdir(model_path) or not os.listdir(model_path)):
    os.makedirs(model_path + '/logs')
    os.makedirs(model_path + '/models')
    os.makedirs(model_path + '/history')
    os.makedirs(model_path + '/figures')
    os.makedirs(model_path + '/params')
    os.makedirs(model_path + '/checkpoints')

# Create train list
train_names = os.listdir(train_path)
num_imgs = len(train_names)
idx = np.arange(num_imgs)
train_ids = idx

# Create generators
train_gen = DataGenerator(folder_path=train_path)

# Model Parameters
params = dict()
params['Number of channels'] = n_channels
params['Batch Size'] = batch_size
params['Epochs'] = epochs
params['Learning rate'] = learning_rate
params['Training split'] = train_split
params['Validation split'] = validation_split
params['Testing split'] = test_split

print(['Model Parameters'])
print('------------')
for key in params.keys():
    print(key + ':', params[key])

# Create Model
SVSRNN = SVSRNN()
model = SVSRNN.get_model()

# Model Summary
print(model.summary())

# Compile Model
model.compile(optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, decay=1e-8),
                loss=['mean_squared_error', 'mean_squared_error'])
callbacks = []

# CSV Logger
callbacks.append(CSVLogger(model_path + '/logs/' + model_name + '.csv'))

# Model Checkpoints
callbacks.append(ModelCheckpoint(model_path + '/checkpoints/' + 'epoch-{epoch:02d}/' + model_name + '.h5', monitor=monitor, save_freq=100))

# Stop on NaN
callbacks.append(TerminateOnNaN())

# Early Stopping
#callbacks.append(EarlyStopping(monitor='loss',min_delta=0.001,patience=15))

# Fit model
start_time = time.time()

print("Starting Training...")
model_history = model.fit(train_gen, 
                                    steps_per_epoch=len(train_ids)//batch_size,
                                    verbose=1, epochs=epochs, callbacks=callbacks)
print("...Finished Training")

elapsed_time = time.time() - start_time

# Save history
with open(model_path + '/history/' + model_name, 'wb') as fp:
    pickle.dump(model_history.history, fp)

# Save parameters
params['Training Times'] = elapsed_time
f = open(model_path + '/params/' + model_name + '.txt', 'w')
f.write('[Model Parameters]' + '\n')
f.write('------------' + '\n')
for k, v in params.items():
    f.write(str(k) + ': '+ str(v) + '\n')
f.close()

timestr = time.strftime('%Y%m%d-%H%M%S.h5')
model.save(model_path + '/models/' + model_name + timestr)
print('Model saved successfully.')

# Display loss curves
fig, ax = plt.subplots(1, 1)
ax.plot(model_history.history['loss'], color='blue', label='Training Loss')
#ax.plot(model_history.history['val_loss'], color='orange', label='Validation Loss')
ax.set_title('Loss Curves')
ax.set_ylabel(model_loss)
ax.set_xlabel('Epochs')
plt.legend()

# Save figure
plt.savefig(model_path + '/figures/' + model_name + '.png')
print('Loss figure saved successfully.')