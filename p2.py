import numpy as np
import pandas as pd
from itertools import combinations_with_replacement        #to get all the terms for a given degree of poly..!

##some helper functions..!!
def mean_squared_error(Y_label, Y_predicted):
    error_vector = Y_label - Y_predicted
    #swuared vector is just the error vector, but with each element squared.. so its
    #actually just element wise multiplication of err_vec with itself.
    squared_vector = error_vector * error_vector

    #np.mean() can also calculate mean of multi dim vecs..but our input is a column vec..
    #so it just gives the arithmetic mean of all the elements present in the vec..!
    return np.mean(squared_vector)

#R^2 metric is basically = 1 - (SS_residual / SS_total), where SS = sum of squares
def r_squared(Y_label, Y_predicted):
    Y_mean = np.mean(Y_label)

    #calculate SS_residual
    ss_res = np.sum((Y_label - Y_predicted) * (Y_label - Y_predicted))

    #calculate SS_total
    ss_tot = np.sum((Y_label - Y_mean) * (Y_label - Y_mean))

    return (1 - (ss_res / ss_tot))



#read the csv into the data frame..!!
#print(dataframe_train), so this dataframe_train basically contains 7 data colums, x1 through x6 and y
dataframe = pd.read_csv("IMT2024003_train_var2.csv")  

# Shuffle the dataframe..! Not sure how this works.. but its better if we shuffle to eleminate any input bias..!
dataframe = dataframe.sample(frac=1, random_state=67).reset_index(drop=True)

#iloc arguments => [row_positions, column_positions]
#syntax is very similar to the how range(a:b:c) works..!
raw_data = dataframe.iloc[::, 0:3].values       #take all rows but only 0-5 columns..!

#now get the labels..!, i.e Y
Y_raw = dataframe.iloc[::, 3:4].values              #a numpy nd array..!

#N = no of samples, d = no of parameters..!
N_total, d = raw_data.shape


#start with an array that is just a column of ones..! and keep horizontally stacking
#all the possible combinations of the columns..!!
temp_nd_array = np.ones((N_total, 1))
best_r_sqr_val = (-1) * 1e9

for degree in range(1, 21):
    feature_indices = list(range(d))

    combinations = list(combinations_with_replacement(feature_indices, degree))

    #make the new X_raw with the appropriate parameters (according to the degree..!) each combination is a tuple..!
    for combination in combinations:
        #slicing with a tuple grabs all those columns at once
        selected_columns = raw_data[:, combination]
        
        #np.prod multiplies all the elements across the given axis, here columns..!
        next_column_to_add = np.prod(selected_columns, axis=1)
        #assuming out training data is a fixed, 1000 X 6 vector ..!!
        next_column_to_add = next_column_to_add.reshape(1000, 1)
        
        # Now stack it safely onto your expanding matrix
        temp_nd_array = np.hstack((temp_nd_array, next_column_to_add))

    print(temp_nd_array.shape)

    X_raw = temp_nd_array

    ###Now make the train / validation split..! (80 : 20)
    split_idx = int(0.8 * N_total)

    X_train = X_raw[:split_idx]
    X_val = X_raw[split_idx:]

    Y_train = Y_raw[:split_idx]
    Y_val = Y_raw[split_idx:]

    #we know, w = (X^T X)^-1 * X^T * y and "(X^T X)^-1 * X^T" is just the pseudo inverse of X..!!
    w = np.linalg.pinv(X_train) @ Y_train

    #now make the predictions..!! Y_hat = X_test * w
    Y_hat = X_val @ w

    #calculate the metrics..!
    mse = mean_squared_error(Y_val, Y_hat)
    r_sqrd = r_squared(Y_val, Y_hat)

    #keep track of the most accurate model so far..!!
    if(r_sqrd > best_r_sqr_val):
        best_degree = degree
        best_weights = w
        best_r_sqr_val = r_sqrd


    print(f"polynomial degree = {degree}, MSE = {mse}, R^2 = {r_sqrd}")


print(f"best_degree = {best_degree}, \nbest_weights = {w}, \nbest_r_sqrd_value = {best_r_sqr_val}")

#load the test dataset
df_test = pd.read_csv("IMT2024003_test_var2.csv")

#extract features (x1 to x6)
raw_test_data = df_test.iloc[:, 0:3].values
N_test_total = raw_test_data.shape[0]

#initialize the test matrix with the bias column
X_test_expanded = np.ones((N_test_total, 1))

#expand features using the optimal degree 
for degree in range(1, best_degree + 1):
    combinations = list(combinations_with_replacement(list(range(d)), degree))
    
    for combination in combinations:
        selected_columns = raw_test_data[:, combination]

        #multiply and dynamically reshape, i.e put -1 as the no of rows..
        #cuz this might be tested on other data and it may not have 1000 rows (like out current test data)..!
        next_column_to_add = np.prod(selected_columns, axis=1).reshape(-1, 1)
        X_test_expanded = np.hstack((X_test_expanded, next_column_to_add))

#make the predictions, on the test data
Y_test_predictions = X_test_expanded @ best_weights

#create a new df, with the column 'y' like in the sample submission
prediction_df = pd.DataFrame(Y_test_predictions, columns=['y'])

# Export to CSV exactly as required by the assignment deliverables
prediction_df.to_csv("IMT2024003_pred_var2.csv", index=False)
