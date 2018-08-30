from moleimages import MoleImages #Helper method
import glob
import os
import numpy as np

'''
This function resizes the images from the benign and malignant folders consecutively
It has a high verbosity, telling what it is currently resizing and what the image # is
'''
def resize_images():
    print('Resizing Benign')
    moles = MoleImages('Benign/*.jpg')
    #moles.resize_bulk("b")
    benigns = moles.resize_bulk("b")
    moles.save_png(benigns, 'data_scaled/benign', tag='bimg-')

    print('Resizing Malign')
    moles = MoleImages('Malignant/*.jpg')
    #moles.resize_bulk("m")
    malignants = moles.resize_bulk("m")
    moles.save_png(malignants,'data_scaled/malign', tag='mimg-')

# GETTING TEST IMAGES WITH A 10% RATIO 
# NOTE THAT THE TEST IMAGES COME FROM THE SAME DIRECTORY AS TRAINING IMAGES
def cv_images(dir_b='data_scaled_validation/benign', dir_m='data_scaled_validation/malign', pct=0.1):
    image_b = glob.glob(os.path.join(os.getcwd(),'data_scaled/benign/*.png')) #Benign folder once scaled
    image_m = glob.glob(os.path.join(os.getcwd(),'data_scaled/malign/*.png')) #Malignant folder once scaled
    n_images_b = int(pct*len(image_b)) # number of images in benign which we want
    n_images_m = int(pct*len(image_m)) # number of images in malignant which we want
    image_b = np.random.choice(image_b,n_images_b, replace=False) # Randomizes benign images which we pick
    image_m = np.random.choice(image_m,n_images_m, replace=False) # Randomizes malignant images which we pick
    
    for img in image_b: #For each benign image
        filename = img.split('/')[-1] #Get the filename
        print('Moving {} to {}'.format(img,dir_b + '/' + filename)) #Print that we are moving that file to the scaled folder
        current_dir = os.getcwd()
        current_dir = (os.path.join(current_dir, dir_b))
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        os.rename(img,current_dir + '/' + filename)
        
    for img in image_m: #Same thing we did for the benign images, now for the malignant images
        filename = img.split('/')[-1]
        print('Moving {} to {}'.format(img,dir_m + '/' + filename))
        
        current_dir = os.getcwd()
        current_dir = (os.path.join(current_dir, dir_m))
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        os.rename(img,current_dir + '/' + filename)

#Main method - first resizing, then moving the images
if __name__ == '__main__':
    ## ESSENTIALLY MAKING SURE WE DO NOT MAKE DUPLICATES OR GET ERRORS
    
    scaled_exists = os.path.exists(os.path.join(os.getcwd(), 'data_scaled'))
    validation_exists = os.path.exists(os.path.join(os.getcwd(), 'data_scaled_validation'))
    print(scaled_exists)
    print(validation_exists)
    if (scaled_exists and validation_exists):
        print("Scaling and validation images generation already complete")
    elif scaled_exists:
        print("Scaled images exist, but validation images do not. Please delete the scaled images folder and restart")
    elif validation_exists:
        print("Validation images exist, but scaled images do not. Please delete the validation images folder and restart")
    else:
        resize_images()
        cv_images()
    
    