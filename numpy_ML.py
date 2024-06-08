import numpy as np
import random
# numpy has onli one data structures i.e. n-d arrays
arr = np.array([[1,2,3],[4,5,6]])
print("\nn-d array using numpy :\n",arr)
print("\n",type(arr))

arr1 = np.zeros((3,4))
print("\nn-d arr with zeros as elmnts :\n",arr1)

arr2 = np.ones((4,2))
print("\nn-d arr with ones as elmnts :\n",arr2)

arr3 = np.identity(4)  #or np.eye(4) would hve given same results
print("\nn-d identity matrix :\n",arr3)

arr4 = np.arange(4,24,3)
print("\narray within a given range :\n",arr4)

arr5 = np.linspace(3,8,21)
print("\narr in range with linearly spaced elmnts :\n",arr5)

arr6 = np.copy(arr3)
print("\ncopied array :\n",arr6)
print("\nattributes in numpy :")
print("1) its dimensions :\n",arr6.ndim)
print("\n2) its shape :\n",arr6.shape)
print("\n3) num of elmnts in the array :\n",arr6.size)
print("\n4) size of each elment(eg : float=8,int=4,etc) :\n",arr6.itemsize)
print("\n5) data type of array :\n",arr6.dtype)
print("\n6)type conversion into another data type : \n",arr6.astype(int)) 
#while data cleaning we have to reduce the footprint of the data this can be used then to convert float to int to save space when not needed

arr7 = np.random.randint(low=20,high=250,size=15)
print("\nrandomly generated array :\n",arr7)
# np.random.seed(1) , this step keeps generating the same random value
# np.random.random()

# python list <<< numpy arrays in terms of speed of execution and saves hell lotta space
a = np.arange(1,7)
b = np.arange(8,14)
c = a+b
print("\nadding two numpy arr :") # others operations like *\-\%\/
print("a :",a,"\tb :",b,"\tadded array(a+b) :",c)
print("\nchecking if elmnts in b is greater than 10 :",b>9)
print("printing those elements which are true :",b[(b>9) & (b%2==0)]) #index using boolean array or filtering
b[(b>9) & (b%2==0)] = 0
print("printing the arr after removing the required true values :",b)

print("\nreshaped arr(a) :\n",a.reshape(3,2),"\n\n(OR)\n\n",a.reshape(2,3))

# array indexing nd slicing
f = np.arange(2,26).reshape(6,4)
print("\narr f : \n",f[:],"\nslicing colmn 1-2 : \n",f[:,1:3],"\nslicing 4th row : \n",f[3],"\nspecific slicing : \n",f[3:5,2:4])
# syntax :- a [:,4] means everything in that row in 4th column , [3,:] means in 3rd row all columns content , [:] means entire matrix
print("accessing random discontinous set of rows of matrix which cannot be sliced : \n",f[[0,3,5]]) #fancy indexing

print("\nfor performing dot product b/w 2 matrices the sytnax is : a.dot(b)\n")

print("the min elmnt of f is :",f.min(),"and the max elmnt is :",f.max()) #np.argmax() gives the index of the maximum element
# axis=0 for column-wise , axis=1 for row-wise operation
print("min elmnt of each row :",f.min(axis=1),"and min elmnt of each column :",f.min(axis=0))
print("max elmnt of each row :",f.max(axis=1),"and max elmnt of each column :",f.max(axis=0))
print("sum of all elmnts:",f.sum(),", sum of rows:",f.sum(axis=1),", sum of colmns:",f.sum(axis=0))
print("\nmean :",f.mean(),", standard deviation :",f.std(),", median :",np.median(f),"\nsine operation :\n",np.sin(f))
# here the way of accesing "mean" and "median" is different due to concept of OOPs :
# mean is normal function inside the class which requires an object in-order to be accessed || Unary operations = min,max,sum,mean,std
# but median is a static function which can be accessed directly   ||  Universal functions = exp,sqrt,log,sin,cos

# iteration
h = np.arange(4,10).reshape(3,2)
k = np.arange(6).reshape(3,2)
print("\nwill print row by row using iteraton : ")
for i in k:
    print("rows :\n",i)
print("\nprints all items one-by-one using loop :\n")
for j in np.nditer(k):  # used when , if say we gotta apply a filter to every pixel in the picture
    print(j)

# changing shape in arr ravel , reshape , transpose , stacking , splitting
print("\nchanges the higher order dimension of the array to one-dimension :",k.ravel())
print("reshape from 3/2 to 2/3 : \n",k.reshape(2,3))
print("transpose of the matrix : \n",k.transpose())
print("stacking matrices horizontally :\n",np.hstack((k,h)))
print("stacking matrices horizontally :\n",np.hstack((h,k)))
print("stacking matrices vertically :\n",np.vstack((k,h)))
print("splitting the array in 3 parts vertically : \n",np.vsplit(k,3))  # vertical split limit is the max number of rows of a matrix
print("\nsplitting the array in 2 parts horizontally : \n",np.hsplit(k,2)) # horizontal split limit is the max num of columns of a matrix

# How to add a border (filled with 0's) around an existing array? (★☆☆)
print("\n")
Z = np.ones((5,5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print("\n",Z)
     # (OR)
# Using fancy indexing
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print("\n",Z)

# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
Z = np.diag(1+np.arange(4),k=-1)
print("\n",Z)

Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z) # checkerboard design


import matplotlib.pyplot as plt
f = np.linspace(0, 2 * np.pi, 100)  
b = np.sin(f)

plt.plot(f, b) # 1st argument is x-axis , 2nd argument is y-axis
plt.xlabel('f')
plt.ylabel('sin(f)')
plt.title('Plot of sin(f)')
plt.show()

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot([1, 2, 3], [4, 5, 6])
ax1.set_title('Plot 1')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')

ax2.plot([4, 5, 6], [7, 8, 9])
ax2.set_title('Plot 2')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')

plt.show()

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

# Plot some data on each subplot
ax1.plot([1, 2, 3], [4, 5, 6])
ax2.plot([1, 2, 3], [6, 5, 4])

# Add labels and titles
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')

plt.suptitle('Two Subplots on Same Figure')

# Display the plot
plt.show()

# np.where(condition , statement excuted when true , statement to be executed when false)
# np.where(a%2!=0, 0 , a)    ---------->  eg