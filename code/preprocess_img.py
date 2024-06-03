import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

def scale_radius(img,scale):
    x=img[img.shape[0]//2, :, :].sum(1)
    r=(x>x.mean()/10).sum()//2
    s=scale*1.0/r
    return cv2.resize(img,(0,0),fx=s,fy=s)



def preprocess_and_plot_image(f,scale=300,output_dir="preprocessed_images"):
    try:
        original_image = cv2.imread(f,cv2.IMREAD_UNCHANGED) # read the image
        if original_image.dtype != np.uint8:
            if original_image.dtype == np.float64: 
                original_image = (original_image*255).astype(np.uint8) # convert the image to uint8 if it is not already
            else:
                original_image = cv2.convertScaleAbs(original_image) # convert the image to uint8 if it is not already
        
        
        processed_image=scale_radius(original_image,scale) # scale the radius of the image
        processed_image = cv2.addWeighted(processed_image,4,cv2.GaussianBlur(processed_image,(0,0),scale/30),-4,128)
        # the above line is a way to sharpen the image by adding a weighted sum of the image and a gaussian blurred version of the image 
        # the gaussian blurred version is subtracted from the original image with a weight of 4 and the result is added to 128
        
        
        mask = np.zeros(processed_image.shape,dtype =np.uint8) # create a mask of zeros with the same shape as the processed image
        
        
        cv2.circle(mask,(processed_image.shape[1]//2,processed_image.shape[0]//2),int(scale*0.9),(1,1,1),-1,8,0) 
        # create a circle in the mask with the same center as the image and a radius of 0.9 times the scale 
        
        
        processed_image = processed_image*mask+128*(1-mask) # apply the mask to the processed image

        output_path = os.path.join(output_dir,os.path.basename(f)) # create the output path
        os.makedirs(output_dir,exist_ok=True) # create the output directory if it does not exist
        cv2.imwrite(output_path,processed_image) # write the processed image to the output path
        # plot the original and processed images
        plot_images(original_image,processed_image,f)
    
    except Exception as e:
        print(f"Error processing file {f}: {e}")



def plot_images(original_image,processed_image,f,output_dir="preprocessed_images"):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1) # create a subplot with 1 row and 2 columns and plot the original image what are the three numbers ofr? 
    # the first number is the row number, the second number is the column number and the third number is the plot number
    
    
    
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)) 
    plt.title("Original Image")
    plt.axis("off") # turn off the axis of the plot

    plt.subplot(1,2,2) # create a subplot with 1 row and 2 columns and plot the processed image
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title("Processed Image")
    plt.axis("off") # turn off the axis of the plot

    plt.suptitle(f"Comparison for {os.path.basename(f)}") # set the title of the plot
    plt.savefig(f"{output_dir}/Plt {os.path.basename(f)}") # save the plot as an image in the output directory

for f in glob.glob("../data/train/*.jpeg"): # loop through all the images in the train directory
    preprocess_and_plot_image(f,scale=300,output_dir = 'preprocessed_images') # preprocess and plot the image


