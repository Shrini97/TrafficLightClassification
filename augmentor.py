import cv2
import numpy as np

class augmentor():
	def __init__(self):
		super(augmentor, self).__init__()
		self.augmentation = True
		self.noise_filter = True
		self.augmentation_count=5
		self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		self.noise_mu = 0
		self.noise_sigma = 10
		self.ch_colors = 255.0

		if self.augmentation:
			print("")
			print("using data augmentations , number :",self.augmentation_count)
			print("")
		else :
			self.augmentation_count=1
			print("")
			print("no augmentations performed")
			print("")	
		
	def augment_batch(self,data,gt):
		if self.augmentation:
			augmented_data = data.copy()
			augmented_gt = gt.copy()
			for i in range(self.augmentation_count):
				new_data = np.zeros((data.shape),dtype="uint8")
				for j in range(data.shape[0]):
					new_data[j,:,:,:] = self.augmentation_map(i,data[j,:,:,:])
				augmented_data.append(new_data,axis=0)
				augmented_gt.append(gt,axis=0)

			flipped_data = augmented_data.copy()
			flipped_gt = augmented_gt.copy()
			flipped_data = flipped_data[:,:,::-1,:]
			flipped_gt[:,5], flipped_gt[:,3] = flipped_gt[:,3], flipped_gt[:,5]
			return self.add_gauss_noise(augmented_data),augmented_gt
		else:
			return self.add_gauss_noise(data),gt

	def add_gauss_noise(self,data):
		if self.noise_filter:
			gauss = np.random.normal(self.noise_mu, self.noise_sigma, data.shape)
			noisy = np.add(gauss,data)
			data = noisy.clip(0,self.ch_colors)
			return data
		else:
			return data		

	def augmentation_map(self,index,img):
		if index==0:
			return self.clahe_hist(img)
		if index==1:
			return self.norm_hist(img)
		if index==2:
			return self.gamma_transformation(img,gamma=1.3)
		if index==3:
			return self.gamma_transformation(img,gamma=0.7)		
		else:
			return img

	def gamma_transformation(self,img,gamma):
		Lab=cv2.split(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB))
		Lab[0] = ((Lab[0]/self.ch_colors)**gamma)*self.ch_colors
		Lab = np.array(Lab).astype('uint8')
		gamma_corrected = cv2.cvtColor(cv2.merge(Lab),cv2.COLOR_LAB2BGR)
		return gamma_corrected

	
	def clahe_hist(self,img):
		#print(img.shape,img.dtype)
		Lab=cv2.split(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB))
		Lab[0]=self.clahe.apply(Lab[0])
		clahe = cv2.cvtColor(cv2.merge(Lab),cv2.COLOR_LAB2BGR)
		return clahe

	def norm_hist(self,img):
		Lab=cv2.split(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2LAB))
		Lab[0]=cv2.equalizeHist(Lab[0])
		equ = cv2.cvtColor(cv2.merge(Lab),cv2.COLOR_LAB2BGR)
		return equ