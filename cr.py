# Import necessary libraries
from skimage import io  # import skimage library to read and display images
import random  # import random library to generate random numbers
import numpy as np  # import numpy library to work with arrays and matrices
import numpy.matlib  # import numpy.matlib library to work with matrices
import scipy.misc  # import scipy.misc library to save compressed images
import os  # import os library to get file information

# Load the image using skimage and display it
image = io.imread('tiger.png')
io.imshow(image)  # display the image
io.show()  # show the image

# Get the number of rows and columns in the image
rows = image.shape[0]
cols = image.shape[1]

# Normalize the image pixel values to the range 0-1
image = image/255

# reshape image to be a 2D array where each row represents a pixel and each column represents a color channel
X = image.reshape(image.shape[0]*image.shape[1], 3)

# Set the number of clusters and the maximum number of iterations for k-means
K = 16  # number of clusters
max_iters = 50  # number of times the k-mean should run

# Define a function to randomly initialize the centroids


def init_centroids(X, K):
    c = random.sample(list(X), K)
    return c

# Define a function to assign each pixel to its nearest centroid


def closest_centroids(X, c):
    K = np.size(c, 0)  # get the number of centroids
    # create an empty array to store centroid assignments
    idx = np.zeros((np.size(X, 0), 1))
    # create an empty array to store distances to centroids
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]  # get the i-th centroid
        # create an array where each row is the i-th centroid
        temp = np.ones((np.size(X, 0), 1))*y
        # calculate the square of the differences between each pixel and the i-th centroid
        b = np.power(np.subtract(X, temp), 2)
        # calculate the sum of the squared differences along each row
        a = np.sum(b, axis=1)
        a = np.asarray(a)  # convert the result to a numpy array
        a.resize((np.size(X, 0), 1))  # resize the array to be a column vector
        # append the column vector to the distance array
        arr = np.append(arr, a, axis=1)
    # delete the first column of the distance array
    arr = np.delete(arr, 0, axis=1)
    # find the index of the closest centroid for each pixel
    idx = np.argmin(arr, axis=1)
    return idx

# Define a function to compute the new centroids based on the mean of the pixels in each cluster


def compute_centroids(X, idx, K):
    n = np.size(X, 1)  # get the number of color channels
    # create an empty array to store the new centroids
    centroids = np.zeros((K, n))
    for i in range(0, K):
        ci = idx == i  # find the pixels that are assigned to the i-th centroid
        ci = ci.astype(int)  # convert the boolean array to an integer array
        # get the total number of pixels assigned to the i-th centroid
        total_number = sum(ci)
        ci.resize((np.size(X, 0), 1))  # resize the array to be a column vector
        # Create a matrix where each row is ci, so that we can use it to multiply by X
        total_matrix = np.matlib.repmat(ci, 1, n)
        ci = np.transpose(ci)
        # Multiply each data point by ci to get a matrix where only the data points assigned to this centroid have nonzero values
        total = np.multiply(X, total_matrix)
        # Compute the new centroid by taking the mean of the data points assigned to this centroid
        centroids[i] = (1/total_number)*np.sum(total, axis=0)
    return centroids

# Define a function to run k-means


def run_kMean(X, initial_centroids, max_iters):
    # Get the number of rows and columns in the input data
    m = np.size(X, 0)
    n = np.size(X, 1)
    # Get the number of initial centroids
    K = np.size(initial_centroids, 0)
    # Initialize the centroids with the given initial centroids
    centroids = initial_centroids
    previous_centroids = centroids
    # Initialize an array to store the indices of the closest centroids to each data point
    idx = np.zeros((m, 1))
    # Loop over the maximum number of iterations to update the centroids and indices
    for i in range(1, max_iters):
        # Assign each data point to its closest centroid
        idx = closest_centroids(X, centroids)
        # Recompute the centroids based on the new assignments
        centroids = compute_centroids(X, idx, K)
    # Return the final centroids and indices
    return centroids, idx


# Initialize the centroids randomly
initial_centroids = init_centroids(X, K)

# Run k-means on the pixel data
centroids, idx = run_kMean(X, initial_centroids, max_iters)

# Assign each pixel to its nearest centroid
idx = closest_centroids(X, centroids)

# Replace each pixel with its corresponding centroid
X_recovered = centroids[idx]

# Reshape the 2D pixel array back into an image
X_recovered = np.reshape(X_recovered, (rows, cols, 3))

# Save the compressed image to disk and display it
scipy.misc.imsave('tiger_small.jpg', X_recovered)

image_compressed = io.imread('tiger_small.jpg')
io.imshow(image_compressed)
io.show()

# Print the size of the original and compressed images in KB
info = os.stat('tiger.png')
print("size of image before running K-mean algorithm: ", info.st_size/1024, "KB")
info = os.stat('tiger_small.jpg')
print("size of image after running K-mean algorithm: ", info.st_size/1024, "KB")
