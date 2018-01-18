import numpy as np

my_matrix = np.loadtxt(open("reducenum.csv","rb"),delimiter="," , skiprows=1)

reduce = my_matrix[:,0]
num = my_matrix[:,1]

X_train = reduce.reshape((-1,1))
y_train = num.reshape((-1,1))

print(X_train)

print(abs(X_train))
# print(y_train)