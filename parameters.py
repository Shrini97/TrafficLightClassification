import numpy as np
from PIL import Image
import os 
from os.path import exists
from numpy import random
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
import json
import cv2
from codecs import open as co_open
from json import dump as j_dump
from json import load as j_load
from tqdm import tqdm

class params(object):
	def __init__(self):
		#paths to the folders to be used 
		self.data_dir="./data/train/"
		self.test_dir="./data/train/"
		self.result_dir="./results/"
		self.save_dir="./check_point/weights-17-0.12.h5"
		
		
		print("")
		print("")
		print("----------------------------------processing dirpaths----------------------------------------- ")
		print("the training data will be taken from ",self.data_dir)
		print("please make sure the above directories are correct")
		print("----------------------------------------------------------------------------------------------")
		
		#input params
		self.img_channels = 3
		self.img_rows = 200
		self.img_cols = 200
		
		#parameters for training or testing.
		self.batch_size = 5
		self.n_classes = 6
		self.nb_epoch = 20
		self.data_augmentation = True
		self.TrainGreyscale = False
		self.iterations = 20000
		self.train_test_ratio=0.9

		self.image_format = '.jpg'
		self.json_format = ".json"

		# self.JsonFile="ClassName.json"
		# with open(self.JsonFile) as f:
		# 	self.ClassID=json.load(f)

		
		self.logfile_path = self.result_dir+'/'+'log.csv'
		self.early_stopper = EarlyStopping(min_delta=0.001, patience=5)
		self.csv = CSVLogger(self.logfile_path, separator=',', append=False)
		
		self.input_tensor = Input(shape=(self.img_rows, self.img_cols,self.img_channels))

		
		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)
		
		self.check_path = 'check_point/'+'weights-{epoch:02d}-{val_loss:.2f}.h5' 
		self.save_after_epoch = ModelCheckpoint(self.check_path,verbose=1,save_best_only=False,save_weights_only=False, mode='auto',period=1)
		

		self.datagen = ImageDataGenerator(
						featurewise_center=False,  # set input mean to 0 over the dataset
						samplewise_center=False,  # set each sample mean to 0
						featurewise_std_normalization=False,  # divide inputs by std of the dataset
						samplewise_std_normalization=False,  # divide each input by its std
						zca_whitening=False,  # apply ZCA whitening
						rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
						width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
						height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
						horizontal_flip=False,  # randomly flip images
						vertical_flip=False)  # randomly flip images


	def images_and_labels(self,condition="training"):
		if condition == "training":
			X,Y=self.data_array_and_labels(self.data_dir)
			assert X.shape[0] == Y.shape[0]
			
			permutation = np.random.permutation(Y.shape[0])
			X,Y=X[permutation],Y[permutation]
			
			X=preprocess_input(X)
			split_index=int(self.train_test_ratio*Y.shape[0])
			return X[:split_index],Y[:split_index],X[split_index:],Y[split_index:]
		else:
			if condition == "testing":
				X,Y=self.data_array_and_labels(self.test_dir)
				assert X.shape[0] == Y.shape[0]
				print(X.shape,Y.shape)
				permutation = np.random.permutation(Y.shape[0])
				X,Y=X[permutation],Y[permutation]
				X=preprocess_input(X)
				return X,Y
	
	def data_array_and_labels(self,directory_source,id_list=None):		
		print("Data extraction function has been called ")
		data=[]
		label=[]
		base_dir=directory_source
		folders = self.get_folders(base_dir)
		files = self.get_files(folders,self.image_format,self.json_format)
		for i in tqdm(range(len(files))):
			file = files[i]
			labelNow = (self.ReadJson(file[:-len(self.image_format)]+self.json_format))
			if(labelNow["status"]==1):
				img = cv2.imread(file)
				img = cv2.resize(img,(self.img_cols,self.img_rows))
				data.append(img)
				label.append(labelNow["class"])
		
		return np.array(data,dtype=np.float64),np.array(label)

	def get_folders(self,root_folder):
		folders = []
		folders.append(root_folder)
		for (dirpath, dirnames, filenames) in os.walk(root_folder):
			for fold in dirnames:
				folders.append(dirpath + fold + '/')
		return(folders)

	def get_files(self,folders,ext,sub_ext):
		files = []
		for folder in folders:
			files_fold = ([(folder+i) for i in os.listdir(folder) if i.endswith(ext)])
			for file in files_fold:
				if(exists(file[:-len(ext)]+sub_ext)):
					files.append(file)
		return(files)

	def data_array_and_labels_old(self,directory_source,id_list=None):		
		print("Data extraction function has been called ")
		data=[]
		label=[]
		base_dir=directory_source
		if id_list==None:
			class_folders_list=[name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
		else:
			print (id_list)
			class_folders_list=map(str,id_list)
			print (class_folders_list)
		for class_name in class_folders_list:
			class_dir=base_dir+class_name+'/'
			image_list=[file for file in os.listdir(class_dir) if file.endswith(self.image_format)]
			for image_name in image_list:
				if self.TrainGreyscale:
					x= Image.open(class_dir+image_name).convert('L')
					im=np.asarray(x)
					data.append(np.array([im,im,im]))
					
				else:
					x= Image.open(class_dir+image_name)
					im=np.asarray(x)
					data.append(np.array(im))
				
				lab=[0]*self.n_classes
				lab[int(class_name)]=1
				label.append(lab)	
		
		if self.TrainGreyscale:
			return np.array(data,dtype=np.float64).transpose(0,2,3,1),np.array(label)
		else:
			return np.array(data,dtype=np.float64),np.array(label)
				
	def ReadJson(self,FilePath):
		InputFile = open(FilePath)
		JsonArray = j_load(InputFile)
		return JsonArray

	def WriteJson(self,dict_,file_path):
		j_dump(dict_, co_open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
