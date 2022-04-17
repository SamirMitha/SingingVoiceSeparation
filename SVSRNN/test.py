import glob, os
import shutil

data_path = '/media/samir/Secondary/Datasets/DSD100/Mixtures/Test'
new_folder = 'test_samples_wav'


folders = os.listdir(data_path)

for folder in folders:
	old_place = data_path + '/' + folder + '/mixture.wav'
	new_place = new_folder + '/' + folder + '.wav'
	shutil.copy(old_place, new_place)