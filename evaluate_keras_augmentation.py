# 写一个展示 Data Augmentation 效果的程序
#  尤其对于大规模的CNN网络，需要进一步重视和评估 Data Augmentation 的作用
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import matplotlib.pyplot as plt
from  PIL import Image
import numpy as np
img1 = Image.open('3.jpg')
img2 =np.asarray(img1)
img3 = img2.reshape(1,img2.shape[0],img2.shape[1],img2.shape[2])
datagen = ImageDataGenerator(
 channel_shift_range = 0.5
)

x = datagen.flow(img3, np.array([1]) , batch_size = 1)
changed_img = x[0][0].reshape(x[0][0].shape[1], x[0][0].shape[2], x[0][0].shape[3])
changed_img = changed_img.astype(np.uint8)
plt.subplot(1,2,1)
plt.imshow(img2)
plt.subplot(1,2,2)
plt.imshow(changed_img)