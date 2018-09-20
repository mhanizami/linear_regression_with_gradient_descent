# Multivariate Linear Regression With Gradient Descent
import csv
def ft_scaling(matrix):
    #Normalizing Algorithm
    return matrix
def mvlrgd_train(matrix,m,n,a):
    i = 0 #Iteration Counter Holder for all training examples m
    j = 0 #Iteration Counter Holder for all features n
    temp = 0.0
    hyp = 0.0
    X = matrix
    c = 100000
    # Initializing T:
    for j in range(n):
        if j == 0:
            T = [0.00]  # Will Grow to be nx1
        else:
            T.append(0.00)
    j = 0
    # Initializing T_old:
    for j in range(n):
        if j == 0:
            T_old = [0.00]  # Will Grow to be nx1 too
        else:
            T_old.append(0.00)
    j = 0
    # Initializing sum_T:
    for j in range(n):
        if j == 0:
            sum_T = [0.00]  # Will grow to be nx1 too
        else:
            sum_T.append(0.00)
    j = 0
    # Initializing Y:
    for i in range(int(m)):
        if i == 0:
            Y = [1.00]  # Will Grow to be mx1
        else:
            Y.append(matrix[i][0])
    i = 0
    # Initializing X:
    for i in range(int(m)):
        X[i][0]=1.00
    i = 0
    # Counting c iterations
    while c >= 0:
        while i < m:
            while j < n:
                temp = temp+T_old[j]*X[i][j]
                j += 1
            hyp = temp
            temp = 0
            j = 0
            while j < n:
                sum_T[j] = sum_T[j]+(hyp-Y[i])*X[i][j]
                j += 1
            j = 0
            while j < n:
                T[j] = T_old[j] - (a/m)*sum_T[j]
                j += 1
            j = 0
            while j < n:
                T_old[j] = T[j]
                j += 1
            j = 0
            i += 1
        i = 0
        c -= 1
    return T
def mvlrgd_predict(T,key_in):
    out = 0.00
    n = len(key_in)
    for j in range(n):
        out = out + T[j]*key_in[j]
    j = 0
    return out
####
a = 0.0001
j = 0
choice = 1
####
with open('E:\Python Files\MvLRwGD\dataset_stline_slope_pt5.csv') as csv_file:   #csv file reading
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            matrix=[list(map(float, row))]
            line_count += 1
            n = len(row)
        else:
            matrix.append(list(map(float, row)))
            line_count += 1
    #x = list(map(float, x))     #convert string to float type
    #y = list(map(float, y))     #convert string to float type
    #print(matrix)
    m = float(line_count)
    print(f'Total {line_count} Examples and {n-1} feature(s)')
####
matrix = ft_scaling(matrix)
print('Learning...Please Wait')
T = mvlrgd_train(matrix,m,n,a)
print('The Learned Parameter Matrix is: ',T)
for j in range(n):
    if j == 0:
        key_in = [1.00]  # Will Grow to be nx1
    else:
        key_in.append(float(input('Enter Next Input Value: ')))
print(mvlrgd_predict(T,key_in))