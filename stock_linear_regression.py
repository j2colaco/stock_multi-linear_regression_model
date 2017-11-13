import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timedelta
import pandas as pd
import pandas_datareader.data as web


def compute_cost(x,y,theta):

    J= 0
    m = y.shape[0]
    h_x = x.dot(theta)
    J = (1/(2*m))*np.sum(np.square(h_x - y))
    return J

def gradient_descent(x,y,theta, alpha, num_iter):
    m = y.shape[0]
    J_array = np.zeros(num_iter)

    for i in range(0, num_iter, 1):
        # print(i)
        h_x = x.dot(theta)
        theta = theta - alpha*(1/m)*x.T.dot((h_x - y))
        J_array[i] = compute_cost(x,y,theta)

    return theta, J_array

if __name__ == '__main__':

    years_of_data = 3

    # Ensures that only Mon-Fri dates are used
    weekday = dt.datetime.today().weekday()
    if (weekday == 5):
        end = dt.datetime.now()- timedelta(days=1)
    elif (weekday == 6):
        end = dt.datetime.now() - timedelta(days=2)
    else:
        end = dt.datetime.today().date()

    start = dt.datetime(int(dt.datetime.today().year - years_of_data), int(dt.datetime.today().month),int(dt.datetime.today().day)).date()
    df = web.DataReader("WPK.TO", 'yahoo', start, end)
    # print(df)
    data = np.array(df)
    # print(data)
    x_temp = data[:,3]
    y = data[:,3]
    y = y[2:]
    predict_y = y[len(y)-3:]
    print(predict_y)
    # print(y)
    # print(x)
    # print(len(x))
    # print(x[len(x)-2])

    x = []
    i = True
    count = 0
    while (i == True):
        if count+1 < len(x_temp)-2:
            x.append([1, x_temp[count], x_temp[count+1]])
            count +=1
        elif count+1 == len(x_temp)-2:
            x.append([1, x_temp[count], x_temp[count + 1]])
            i = False
    # print(x)
    x = np.array(x)
    theta = [0,0,0]

    # print(len(x), len(y))
    #
    # cost = compute_cost(x,y,theta)
    # print(cost)
    num_iter = 1500
    theta, J_array = gradient_descent(x,y,theta, 0.0001, num_iter)
    # print(J_array)
    print(theta)

    haha = predict_y.dot(theta)
    print(haha)
    # x = np.c_[np.ones(data.shape[0]), data[:,0]]
    # y = data[:,1]
    #
    # # plt.scatter(x[:,1], y)
    # # plt.show()
    #
    # theta = [0,0]
    # # cost = compute_cost(x,y,theta)
    # num_iter = 1500
    # theta, J_array = gradient_descent(x,y,theta, 0.01, num_iter)
    #
    # plt.plot(J_array)
    # # plt.show()

