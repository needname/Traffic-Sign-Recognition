import os
import pickle
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
import warnings
from load_pickled_data import load_pickled_data
warnings.filterwarnings("ignore")

#------------------------------------------------------------------

X_train, y_train = load_pickled_data(os.getcwd()+"/training.p", ['features', 'labels'])
#check dimension
n_train = y_train.shape[0]  #return 37** images
image_shape = X_train[0].shape #return (32,32,3) size 32x32x3 (3 channels)
# X_train.shape = (3625, 32, 32, 3) with 3625 images
image_size = image_shape[0]
# Find the unique elements of an array in sorted order
# return_index : bool, optional
# If True, also return the indices of ar (along the specified axis, if provided, or in the flattened array)
# that result in the unique array.an image is resize
# return_counts : bool, optional
# If True, also return the number of times each unique item appears in ar
sign_classes, class_indices, class_counts = np.unique(y_train, return_index = True, return_counts = True)
n_classes = class_counts.shape[0]

print("Number of training examples =", n_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#check size and range of rgb values( rescaled in range 0-1)

for image in X_train[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


#----------------------------------------------------------------

#insert an array to an array

def insertArr(X, y, X_in, y_in, labels):
    sign_classes, class_indices, class_counts = np.unique(y, return_index = True, return_counts = True)
    n_classes = class_counts.shape[0]

    for label in labels:
        index = class_indices[label-1]+1

        X = np.insert(X, index, X_in[y_in == label], axis = 0)
        y = np.append(y, y_in[y_in == label], axis = 0)
        y = np.sort(y)
        sign_classes, class_indices, class_counts = np.unique(y, return_index = True, return_counts = True)
    return X, y

#----------------------------------------------------------------

#Display first image of each class
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)
    n_classes = class_counts.shape[0]
    for label in class_indices:
        image = images[label]
        plt.subplot(8,8,i)
        plt.axis('off')
        plt.title("Class {0} ({1})".format(i, class_counts[i-1]))
        i += 1
        plt.imshow(image)
    plt.show()
    
    #Use with list
    '''
    for label in unique_labels:
        # Pick the first image for each label.
        # list.index(obj) returns the lowest index in list that obj appears
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    plt.show()
    '''    

#display_images_and_labels(X_train, y_train)

#----------------------------------------------------------------

#Display all images of a class
def display_label_images(images, labels, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)
    start = class_indices[label-1]
    end = start + class_counts[label-1]
    random_indices = random.sample(range(start, end), limit)
    for index in random_indices[:][:limit]:
        image = images[index]
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

#display_label_images(X_train, y_train, 1)


#----------------------------------------------------------------

#Display random 10 images of each class and numbers of image of each classes
def display_random_images(images, labels):
    signnames = ['name','stop sign','turn left','turn right','no turn left', 'no turn right', 'no entry', 'max speed', 'speed limit']
    col_width = max(len(name) for name in signnames)
    sign_classes, class_indices, class_counts = np.unique(labels, return_index = True, return_counts = True)
    i = 1
    fig = pyplot.figure(figsize = (10, 10))
    for c, c_index, c_count in zip(sign_classes, class_indices, class_counts):
        #PRINT print("%-*s %s" % (8,'noname','pik'))->noname   pik
        print("Class %i: %-*s  %s samples" % (c, col_width, signnames[c], str(c_count)))
        
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
        random_indices = random.sample(range(c_index, c_index + c_count), 10)
        '''
        for i in range(10):
            axis = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
            axis.imshow(X_train[random_indices[i]])
        '''
        for index in random_indices:
            image = images[index]
            plt.subplot(8, 10, i)  # 8 rows, 10 per row
            plt.axis('off')
            i += 1
            plt.imshow(image)
        print("--------------------------------------------------------------------------------------\n")
    plt.show()
    #Display a graph of classes
    #numpy.arange([start, ]stop, [step, ]dtype=None)
    '''
    pyplot.bar( np.arange(1,9,1), class_counts, align='center' )
    pyplot.xlabel('Class')
    pyplot.ylabel('Number of training examples')
    pyplot.xlim([0, 9])
    pyplot.show()
    '''
#display_random_images(X_train, y_train)
#----------------------------------------------------------------

#Preprocess
import numpy as np
from sklearn.utils import shuffle
from skimage import exposure

#-------------------------------------------------------------------

#Augmentation: flip, rotation, projection

#Flip

def flip_extend(X,y):
    # Classes of signs that, when flipped horizontally then vertically should still be classified as the same class
    self_flippable_both = np.array([8])
    # Classes of signs that when flipped horizontally,X_extended = np.append(X_extended, X[y == c], axis = 0) would still be meaningful, but should be classified as a diff class
    cross_flippable = np.array([
        [2,3],
        [3,2],
    ])
    num_classes = 8

    # initial array to contain X_extend, y_extend
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.dtype)

    for c in range(1,num_classes+1):
        # first copy existing data for this class
        X_extended = np.append(X_extended, X[y==c], axis = 0)
        # If we can flip images of this class horizontally and they would belong to other class...
        if c in cross_flippable[:,0]:
            # Copy flipped images of that othe class to the extended array
            # x = range(5)
            # x = [0, 1, 2, 3, 4]
            # x[::-1] = [4, 3, 2, 1, 0] # slice a sequence in reverse  
            flip_class = cross_flippable[cross_flippable[:,0] == c][0][1]
            #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.append.html
            X_extended = np.append(X_extended, X[y == flip_class][:,:,::-1,:], axis = 0)
        # Fill labels added images set to current class
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]),c,dtype = int))

        # If we can flip images of this class horizontally AND then vertically and they would still belong to the same class
        if c in self_flippable_both:
            #Copy their flippable versions into extended array
            X_extended = np.append(X_extended, X_extended[y_extended == c][:,::-1,::-1,:], axis = 0)
        # Fill labels for added images set to current class
        # numpy.full(shape, fill_value, dtype=None, order='C')[source]
        # Return a new array of given shape and type, filled with fill_value
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    return (X_extended, y_extended)

X_flip_extend, y_flip_extend = flip_extend(X_train,y_train)
#display_random_images(X_flip_extend, y_flip_extend)

#-------------------------------------------------------------------

#from nolearn.lasagne import BatchIterator
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
import random

#rotate class 4,5,7,8
def rotate_extended(X, y, intensity, labels):
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.dtype)

    delta = 30. * intensity # scale using augmentation intensity
    
    for c in labels:
        X_temp = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
        X_temp = np.append(X_temp, X[y==c], axis = 0)
        
        for i in range(X_temp.shape[0]):
        # if don't have preserve_range = True-> convert to float[0,1]
            X_temp[i] = rotate(X_temp[i], random.uniform(-delta, delta), preserve_range = True, mode = 'edge')#random.uniform(-delta, delta), preserve_range = True, mode = 'edge')
        X_extended = np.append(X_extended, X_temp,axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    return X_extended, y_extended

labels_rotate = [4,5,7,8]
X_rotate_extend, y_rotate_extend = rotate_extended(X_train, y_train, 0.75, labels_rotate)
#display_random_images(X_rotate_extend, y_rotate_extend)
X_temp, y_temp = insertArr(X_flip_extend, y_flip_extend, X_rotate_extend, y_rotate_extend, labels_rotate)

#project class 4(x11), 5(x10), 7(x1), 8(x1) = 4+7+8+10x(4+5)
def apply_projection_transform_extended(X, y, intensity, labels):
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
    y_extended = np.empty([0], dtype = y.dtype)
    
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity
    for c in labels:
        X_temp = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = X.dtype)
        X_temp = np.append(X_temp, X[y==c], axis = 0)
        
        for i in range(X_temp.shape[0]):
            tl_top = random.uniform(-d, d)     # Top left corner, top margin
            tl_left = random.uniform(-d, d)    # Top left corner, left margin
            bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
            bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
            tr_top = random.uniform(-d, d)     # Top right corner, top margin
            tr_right = random.uniform(-d, d)   # Top right corner, right margin
            br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
            br_right = random.uniform(-d, d)   # Bottom right corner, right margin
            
            #http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.ProjectiveTransform
            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
            X_temp[i] = warp(X_temp[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge', preserve_range = True)
        X_extended = np.append(X_extended, X_temp,axis = 0)
        y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype = int))
    return X_extended, y_extended

labels_projective_1 = [4, 5, 7, 8]
X_projective_extend, y_projective_extend = apply_projection_transform_extended(X_train, y_train, 0.75, labels_projective_1)
X_temp, y_temp = insertArr(X_temp, y_temp, X_projective_extend, y_projective_extend, labels_projective_1)

labels_projective_10 = [4, 5]
for i in range (10):
    X_projective_extend, y_projective_extend = apply_projection_transform_extended(X_train, y_train, 0.75, labels_projective_10)
    X_temp, y_temp = insertArr(X_temp, y_temp, X_projective_extend, y_projective_extend, labels_projective_1)    

X_final = X_temp
y_final = y_temp

#display_random_images(X_projective_extend, y_projective_extend)

display_random_images(X_final, y_final)

train_balanced = { "features": X_final, "labels": y_final }
pickle.dump( train_balanced, open( "training_balanced.p", "wb" ) )

#Check file training_balanced.p
#X_1, y_1 = load_pickled_data(os.getcwd()+"/training_balanced.p", ['features', 'labels'])
#display_random_images(X_1, y_1)

'''
#http://lasagne.readthedocs.io/en/latest/user/tutorial.html
#http://deeplearning.net/software/theano/extending/graphstructures.html
# fix loi theano: https://github.com/josephwinston/VIN/commit/101de34633f45a8b36b020a4472e2d6f835c535b
class AugmentedSignsBatchIterator(BatchIterator):
    """
    Iterates over dataset in batches. 
    Allows images augmentation by randomly rotating, applying projection, 
    adjusting gamma, blurring, adding noize and flipping horizontally.
    """
        
    def __init__(self, batch_size, shuffle = False, seed = 42, p = 0.5, intensity = 0.5):
        """
        Initialises an instance with usual iterating settings, as well as data augmentation coverage
        and augmentation intensity.
        
        Parameters
        ----------
        batch_size:
                    Size of the iteration batch.
        shuffle   :
                    Flag indicating if we need to shuffle the data.
        seed      :
                    Random seed.
        p         :
                    Probability of augmenting a single example, should be in a range of [0, 1] .
                    Defines data augmentation coverage.
        intensity :
                    Augmentation intensity, should be in a [0, 1] range.
        
        Returns
        -------
        New batch iterator instance.
        """
        super(AugmentedSignsBatchIterator, self).__init__(batch_size, shuffle, seed)
        self.p = p
        self.intensity = intensity

    def transform(self, Xb, yb):
        """
        Applies a pipeline of randomised transformations for data augmentation.
        """
        Xb, yb = super(AugmentedSignsBatchIterator, self).transform(
            Xb if yb is None else Xb.copy(), 
            yb
        )
        
        if yb is not None:
            batch_size = Xb.shape[0]
            image_size = Xb.shape[1]
            
            Xb = self.rotate(Xb, batch_size)
            Xb = self.apply_projection_transform(Xb, batch_size, image_size)

        return Xb, yb
        
    def rotate(self, Xb, batch_size):
        """
        Applies random rotation in a defined degrees range to a random subset of images. 
        Range itself is subject to scaling depending on augmentation intensity.
        """
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace = False):
            delta = 30. * self.intensity # scale by self.intensity
            Xb[i] = rotate(Xb[i], random.uniform(-delta, delta), mode = 'edge')
        return Xb   
    
    def apply_projection_transform(self, Xb, batch_size, image_size):
        """
        Applies projection transform to a random subset of images. Projection margins are randomised in a range
        depending on the size of the image. Range itself is subject to scaling depending on augmentation intensity.
        """
        d = image_size * 0.3 * self.intensity
        for i in np.random.choice(batch_size, int(batch_size * self.p), replace = False):        
            tl_top = random.uniform(-d, d)     # Top left corner, top margin
            tl_left = random.uniform(-d, d)    # Top left corner, left margin
            bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
            bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
            tr_top = random.uniform(-d, d)     # Top right corner, top margin
            tr_right = random.uniform(-d, d)   # Top right corner, right margin
            br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
            br_right = random.uniform(-d, d)   # Bottom right corner, right margin

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                    (tl_left, tl_top),
                    (bl_left, image_size - bl_bottom),
                    (image_size - br_right, image_size - br_bottom),
                    (image_size - tr_right, tr_top)
                )), np.array((
                    (0, 0),
                    (0, image_size),
                    (image_size, image_size),
                    (image_size, 0)
                )))
            Xb[i] = warp(Xb[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

        return Xb

X_train, y_train = load_pickled_data("traffic-signs-data/train.p", columns = ['features', 'labels'])
X_train = X_train / 255.

batch_iterator = AugmentedSignsBatchIterator(batch_size = 5, p = 1.0, intensity = 0.75)
for x_batch, y_batch in batch_iterator(X_train, y_train):
    for i in range(5): 
        # plot two images:
        fig = pyplot.figure(figsize=(3, 1))
        axis = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
        axis.imshow(X_train[i])
        axis = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
        axis.imshow(x_batch[i])
        pyplot.show()
    break

#--------------------------------------------------------------------

def extend_balancing_classes(X, y, aug_intensity = 0.5, counts = None):
    """
    Extends dataset by duplicating existing images while applying data augmentation pipeline.
    Number of generated examples for each class may be provided in `counts`.
    
    Parameters
    ----------
    X             : ndarray
                    Dataset array containing feature examples.
    y             : ndarray, optional, defaults to `None`
                    Dataset labels in index form.
    aug_intensity :
                    Intensity of augmentation, must be in [0, 1] range.
    counts        :
                    Number of elements for each class.
                    
    Returns
    -------
    A tuple of X and y.    
    """
    num_classes = 8

    _, class_counts = np.unique(y, return_counts = True)
    max_c = max(class_counts)
    total = max_c * num_classes if counts is None else np.sum(counts)
    
    X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype = np.float32)
    y_extended = np.empty([0], dtype = y.dtype)
    print("Extending dataset using augmented data (intensity = {}):".format(aug_intensity))
    
    for c, c_count in zip(range(num_classes), class_counts):
        # How many examples should there be eventually for this class:
        max_c = max_c if counts is None else counts[c]
        # First copy existing data for this class
        X_source = (X[y == c] / 255.).astype(np.float32)
        y_source = y[y == c]
        X_extended = np.append(X_extended, X_source, axis = 0)
        for i in range((max_c // c_count) - 1):
            batch_iterator = AugmentedSignsBatchIterator(batch_size = X_source.shape[0], p = 1.0, intensity = aug_intensity)
            for x_batch, _ in batch_iterator(X_source, y_source):
                X_extended = np.append(X_extended, x_batch, axis = 0)
                #print_progress(X_extended.shape[0], total)
        batch_iterator = AugmentedSignsBatchIterator(batch_size = max_c % c_count, p = 1.0, intensity = aug_intensity)
        for x_batch, _ in batch_iterator(X_source, y_source):
            X_extended = np.append(X_extended, x_batch, axis = 0)
            #print_progress(X_extended.shape[0], total)
            break
        # Fill labels for added images set to current class.
        added = X_extended.shape[0] - y_extended.shape[0]
        y_extended = np.append(y_extended, np.full((added), c, dtype = int))
        
    return ((X_extended * 255.).astype(np.uint8), y_extended)

X_final, y_final = extend_balancing_classes(X_train, y_train)
display_random_images(X_final, y_final)
'''
