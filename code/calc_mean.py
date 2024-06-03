import os
from PIL import Image
import numpy as np

# load the valid images in a list of numpy array for each image
def load_images(folder):
    images=[]
    count=0
    for filename in os.listdir(folder):
        if filename.endswith(('.png','.jpg','.jpeg')):
            count+=1
            img = Image.open(os.path.join(folder,filename)).convert('RGB')
            img = img.resize((256,256))
            images.append(np.array(img))
        print(count)
    return images

#for the stack of numpy arrays of images, caluclate the mean and standard deviation in the form of RGB.
def calculate_mean_std(images):
    all_images = np.stack(images,axis=0)
    mean = np.mean(all_images,axis=(0,1,2))
    std = np.std(all_images, axis=(0,1,2))
    return mean,std

folder = '../data/fundus_757/train/'
images = load_images(folder)
mean,std = calculate_mean_std(images)

print("Mean in RGB: ",mean)
print("Standard devaition in RGB: ", std)

# for the 35126 images 
# Mean in RGB:  [81.68494431 57.23721553 41.06358987]
# Standard devaition in RGB:  [77.13295033 55.72475488 44.41507843]

# for the 757 images
# Mean in RGB:  [73.15735804 52.41642731 32.08570788]
# Standard devaition in RGB:  [56.52486039 41.72785028 26.89512404]