from __future__ import print_function
from argparse import ArgumentParser
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from parameters import params

from keras.preprocessing import image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import pylab as plt

import numpy as np
import keras
import h5py 
import matplotlib.pyplot as plt
import os
import csv
import scipy.misc
from model import *
import tensorflow as tf

#ros imports 
import cv2
from sensor_msgs.msg import Image
# from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge,CvBridgeError
import rospy
from tqdm import tqdm

bridge = CvBridge()

class Resnet(params):
	def __init__(self,condition):
		super(Resnet, self).__init__()
		self.ClassIndex = self.ReadJson("index.json")
		if condition == "training":
			self.model = Classifier(self.input_tensor,self.n_classes)
			
			for layer in self.model.layers:
		   		layer.trainable = True
		elif(condition == "ros" or condition == "testing"):
			self.model = load_model("./check_point/weights-18-0.32.h5")
			self.model._make_predict_function()
		
	def	train(self):
		"""
		This function as the name suggests is for training the neural net
		there is an augmentation that the user can use if they wish to
		after training is finished as per the traininng finishing criterion if satisfied by 
		early stopper the functions plots the loss curve and val loss curve
		"""
		self.X_train, self.Y_train, self.X_val, self.Y_val= self.images_and_labels("training")
		
		if self.data_augmentation:
			print('Using real-time data augmentation.')
			
			self.datagen.fit(self.X_train)
			print("augmented_data generated. now fitting the model")
			print(self.X_train.shape,self.Y_train.shape)
			self.run_model = self.model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size),steps_per_epoch=self.X_train.shape[0] // self.batch_size,validation_data=(self.X_val, self.Y_val),epochs=self.nb_epoch, verbose=1, max_q_size=100,callbacks=[self.save_after_epoch,self.csv])
		else:
			print('Not using data augmentation.')
			self.run_model = self.model.fit(self.X_train, self.Y_train,batch_size = self.batch_size,nb_epoch = self.nb_epoch,validation_data=(self.X_val, self.Y_val),shuffle=True,callbacks=[self.save_after_epoch,self.csv,self.early_stopper],verbose = 1)
		
		self.plot_all()
		self.model.save(self.save_dir)

	def GenConfMat(self,results,GT):
		"""
		The purpose of this function is to get the statistics of accuracies of the model 
		onn ann unknown test set.
		we generate a confusion mat and a probabibility mat in the form of a heatmap and also 
		a csv file.
		"""
		self.confusion_mat = np.zeros((2**self.n_classes,2**self.n_classes))
		self.classwise_mat = np.zeros((2**self.n_classes,self.n_classes,2))
		self.variance_mat = np.zeros((2**self.n_classes,2**self.n_classes))
		self.avg_prob = np.zeros((2**self.n_classes,self.n_classes))

		ClassCount = [0 for i in range(2**self.n_classes)]
		for i in range(len(GT)):
			ClassNum = self.ClassIndex.index(GT[i])
			ClassCount[ClassNum] += 1

			result = results[i]
			ProbClass  = np.absolute(np.array(self.ClassIndex)*result)
			ProbClass += np.absolute(np.array(1.0-np.array(self.ClassIndex))*(1.0-np.array(result)))
			self.confusion_mat[ClassNum,:] += np.product(ProbClass,axis=1)

			self.classwise_mat[ClassNum,:,1] += result
			self.classwise_mat[ClassNum,:,0] += 1.0 - np.array(result)

			# self.variance_mat[ClassNum,:] = np.sum((np.array(self.ClassIndex)-np.array(result))**2,axis=1)/self.n_classes
			self.avg_prob[ClassNum,:] += result

		ClassAvail = 0
		classname_list = 
		for i in range(2**self.n_classes):
			if(ClassCount[i]):
				ClassAvail += 1
				self.confusion_mat[i] /= ClassCount[i]
				self.classwise_mat[i] /= ClassCount[i]
				self.avg_prob[i] /= ClassCount[i]
				classname = ""
				for j in range(len(self.ClassIndex[i])):
					classname += str(self.ClassIndex[i][j])
				classname_list.append(classname)
				classwise_heat = plt.imshow(self.classwise_mat[i], cmap='hot')
				plt.yticks(np.arange(self.n_classes), ("red solid","yellow solid","green solid","left green","straight green","right green"))
				plt.title(classname + " (" + str(ClassCount[i]) + ")")
				plt.colorbar(classwise_heat, orientation='horizontal')
				plt.savefig(str(self.result_dir + "classwise/" + classname + ".png"))
				plt.close()

		for i in range(len(GT)):
			ClassNum = self.ClassIndex.index(GT[i])
			result = results[i]

			var = np.sum((self.avg_prob - np.array(result))**2,axis=1)
			self.variance_mat[ClassNum,:] += var/(ClassCount[ClassNum])

		self.std_devi = self.variance_mat**0.5

		confusion_heat = plt.imshow(self.confusion_mat[:ClassAvail], cmap='hot')
		plt.yticks(np.arange(ClassAvail), tuple(classname_list[:ClassAvail]))
		plt.colorbar(confusion_heat, orientation='horizontal')
		plt.savefig(str(self.result_dir + 'confusion_mat.png'))
		plt.close()

		std_devi_heat = plt.imshow(self.std_devi[:ClassAvail], cmap='hot')
		plt.yticks(np.arange(ClassAvail), tuple(classname_list[:ClassAvail]))
		plt.colorbar(std_devi_heat, orientation='horizontal')
		plt.savefig(str(self.result_dir + 'std_devi.png'))
		plt.close()

		np.savetxt(self.result_dir+"confusion_mat.csv", self.confusion_mat[:ClassAvail], delimiter=",")
		np.savetxt(self.result_dir+"std_devi.csv", self.std_devi[:ClassAvail],delimiter=",")

	def plot_all(self):
		"""
		This function is called to plot the logs of training the neural net
		the plots generated are training loss , validation loss,train acc annd validationn acc  
		"""
		plt.plot(self.run_model.history['loss'])
		plt.plot(self.run_model.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.result_dir+'model_loss.png')

		plt.plot(self.run_model.history['acc'])
		plt.plot(self.run_model.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig(self.result_dir+'model_accuracy.png')	

	def test(self,conf = False):
		# dataCount = {}
		folders = self.get_folders(self.test_dir)
		files = self.get_files(folders,self.image_format,self.json_format)
		X_test = np.zeros((1,200,200,3),dtype="uint8")

		GT, pred = [],[]
		for z in tqdm(range(len(files))):
			dict_ = self.ReadJson(files[z][:-3]+"json")
			if(dict_["status"]==1):
				ori_img = cv2.imread(files[z])
				X_test[0] = cv2.resize(ori_img,(200,200))
				Y_test = dict_["class"]
				GT.append(Y_test)
				results = self.model.predict(preprocess_input(X_test))
				result = np.array(results[0]).astype("uint8")
				pred.append(results[0].tolist())
				print(z,len(files),Y_test,results[0])
		self.GenConfMat(pred,GT)
		# self.WriteJson(dataCount,"./data.json")
		
	def callback(self,img):
		"""
		this function under the assumption that it receives a 4-D (number of images,width,height,1) sized image
		and use the predict method of the model class 
		the result generated is a (nnumber of images,number of classes) sized probability distribution for
		wach image sample
		"""
		try:
			ros_img = bridge.imgmsg_to_cv2(img, "bgr8")
			if ros_img is not None:
				H,W,C=ros_img.shape
				feed_list=[]
				for i in range(W//self.img_cols):
					"""
					Below mentioned is the equation for the greyscale conversion algorithms as used by the PIL module
					I have kept it the same as to not allow conflict.
					The images are read snippets of 200,200,3 col wise and concerted to greyscale images (200,200,1) which is duplicated along the channel axis to make the 
					input 200,200,3 as this is the shape of the input the Neural net takes
					"""
					cv2.imshow("sign",ros_img)
					cv2.waitKey(1)

					ros_img=ros_img.astype("float64")
					GreyScaleImg=0.114*ros_img[:,i*200:i*200+200,0]+0.587*ros_img[:,i*200:i*200+200,1]+0.299*ros_img[:,i*200:i*200+200,2]
					feed_list.append([GreyScaleImg,GreyScaleImg,GreyScaleImg])

				FeedArray=np.array(feed_list,dtype=np.float64).transpose(0,2,3,1)	
				result_vector=self.model.predict(preprocess_input(FeedArray))
				PredictedIndex=np.argmax(result_vector,axis=1)
				print(PredictedIndex)
				
		except CvBridgeError as e:
			print('this is bad')
			pass

	def ros(self):
		#this function is the listener 
		print('ros successfully called')
		rospy.init_node('image_listener', anonymous=True)
		rospy.Subscriber('/trafficsignimage',Image,self.callback)
		rospy.spin()

parser = ArgumentParser()
parser.add_argument("-m", "--mode", dest="action",default='train')

args = parser.parse_args()


if(args.action=='train'):
	net = Resnet("training")
	net.train()

elif(args.action=='test'):
	net = Resnet("testing")
	net.test()

elif(args.action=='ros'):
	net = Resnet("ros")
	net.ros()