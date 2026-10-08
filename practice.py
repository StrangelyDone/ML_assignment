import numpy as np

#3 rows, 2 columns..! 3 cross 2 shape..!
X = np.array([[1, 2],
              [2, 8],
              [4, 5]
            ])

Y = np.array([6, 7, 8])

print(X)
#print(type(X))             -> A numpy nd array..!

print(f"shape of X is{X.shape}")
#print(type(X.shape))       -> A tuple..!! nice to know..!

#N = no of data points, d = no of dimensions..!
N, d = X.shape

#the shape of the n dim array needs to be passed..!
#and since we know the shape is a tuple..!
ones_column = np.ones((N, 1))
print(ones_column)

#to make the final data matrix, we need to add a column of ones..!
#something like gluing the column of ones to the left of the existing X..!
#we need to pass in the things we need to stick together in a tuple..!
X = np.hstack((ones_column, X))     #other arguments include stuff like dtype etc..
print(X)

#making the transpose
print(X.T)

#matrix multiplication!! use '@'..!
#gram_matrix = (X^T) * (X)
gram_matrix = X @ X.T
print(gram_matrix)

#gram_matrix_inverse = ((X^T) * (X)) ^ -1
#numpy has a specialized sub library called "linalg" (probably linear algebra)..!
#the inverse method is in this sublibrary..! 
gram_matrix_inverse = np.linalg.pinv(gram_matrix)
print(gram_matrix_inverse)

#w = (X^T X)^-1 * X^T * y
w = gram_matrix_inverse @ X.T @ Y
print(f"weights vector = {w}")

