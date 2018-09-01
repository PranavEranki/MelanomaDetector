'''
Author@PranavEranki
'''
import numpy as np
from skimage import io
from skimage.transform import resize
import glob
import h5py
import os


'''
This is the helper method for image prep
'''
class MoleImages():
    def __init__(self, dir=None):
        self.dir = dir
        self.size = None

    # Resizing multiple images to a 128 x 128 size
    def resize_bulk(self, wtype, size=(128,128)):
        '''
        Resize Images and create matrix
        Input: size of the images (128,128)
        Output: Numpy array of (size,num_images)
        '''
        self.size = size
        X = []
        image_list = glob.glob(self.dir) #Getting images we need to resize
        n_images = len(image_list)
        most = 2000
        if n_images > most:
            n_images = most
            image_list = image_list[:n_images]
        # Resizing of the images with verbosity
        print('Resizing {} images:'.format(n_images))
        for i, imgfile in enumerate(image_list):
            print('Resizing image {} of {}'.format(i+1, n_images))
            img = io.imread(imgfile)
            img = resize(img, self.size)
            
            X.append(img)
            
                
        return np.array(X)

    def load_test_images(self, dir_b, dir_m):
        X = []
        image_list_b = glob.glob(os.path.join(os.getcwd(), dir_b + '/*.png'))
        
        n_images_b = len(image_list_b)
        print('Loading {} images of class benign:'.format(n_images_b))
        for i, imgfile in enumerate(image_list_b):
            print('Loading image {} of {}'.format(i+1, n_images_b))
            img = io.imread(imgfile)
            X.append(img)
        image_list_m = glob.glob(os.path.join(os.getcwd(), dir_m + '/*.png'))
        
        n_images_m = len(image_list_m)
        print('Loading {} images of class benign:'.format(n_images_m))
        for i, imgfile in enumerate(image_list_m):
            print('Loading image {} of {}'.format(i+1, n_images_m))
            img = io.imread(imgfile)
            X.append(img)
        
        y = np.hstack((np.zeros(n_images_b), np.ones(n_images_m)))

        return np.array(X), y.reshape(len(y),1)

    def load_image(self, filename, size=(128,128)):
        self.size = size
        img = io.imread(filename) #Getting image
        img = resize(img, self.size, mode='constant') * 255 # Resizing image
        if img.shape[2] == 4: #Making sure everything is 3 channels only
            img = img[:,:,0:3]
        return img.reshape(1, self.size[0], self.size[1], 3)

    def save_h5(self, X, filename, dataset):
        '''
        Save a numpy array to a data.h5 file specified.
        Input:
        X: Numpy array to save
        filename: name of h5 file
        dataset: label for the dataset
        '''
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset(dataset, data=X)
        print('File {} saved'.format(filename))

    def load_h5(self, filename, dataset):
        '''
        Load a data.h5 file specified.
        Input: filename, dataset
        Output: Data
        '''
        with h5py.File(filename, 'r') as hf:
            return hf[dataset][:]

    def save_png(self, matrix, dir, tag='img', format='png'):
        
        # Saving the picture to the directory
        for i, img in enumerate(matrix):
            current_dir = os.getcwd()
            # getting the appropriate filename and directory
            if dir[-1] != '/':
                current_dir = (os.path.join(current_dir, dir + "/"))
                filename = tag + str(i) + '.' + format
            else:
                current_dir = (os.path.join(current_dir, dir))
                filename = tag + str(i) + '.' + format
                
            # this is some verbosity which I implemented for bug testing - not important
            print('Saving file {}'.format(filename))
            print(current_dir)
            
            # Making rhe dir benign / malign for data scaled if not present
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)
                
            # Saving the image to the proper directory
            current_dir = os.path.join(current_dir, filename)
            
            io.imsave(current_dir, img)
            


if __name__ == '__main__':
    pass
    #benign = MoleImages()
    #X = benign.load_h5('benigns.h5','benign')
