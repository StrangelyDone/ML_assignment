import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from sklearn.linear_model import Lasso

#just some helper functions..!
def mean_squared_error(Y_label, Y_predicted):
    return np.mean((Y_label - Y_predicted) ** 2)


def r_squared(Y_label, Y_predicted):
    ss_res = np.sum((Y_label - Y_predicted) ** 2)
    ss_tot = np.sum((Y_label - np.mean(Y_label)) ** 2)
    return 1 - (ss_res / ss_tot)


def soft_threshold(w, lambda_penalty, lr):
    shrinkage = lambda_penalty * lr
    return np.sign(w) * np.maximum(np.abs(w) - shrinkage, 0)


#load and prepare stuff..!
def build_polynomial_design_matrix(raw_data, max_degree=10):
    # make the data matrix for degree 10..!
    N_total, d = raw_data.shape
    X_expanded = np.ones((N_total, 1))
    feature_degrees = [0]

    for degree in range(1, max_degree + 1):
        combinations = list(combinations_with_replacement(list(range(d)), degree))
        for combination in combinations:
            selected_columns = raw_data[:, combination]
            #just put -1 as first param, cuz we could be dealing with any no of data points..!
            next_column = np.prod(selected_columns, axis=1).reshape(-1, 1)
            X_expanded = np.hstack((X_expanded, next_column))
            feature_degrees.append(degree)

    return X_expanded, feature_degrees


def find_best_weights(inputFileName, debug=True):
    #read the csv into the data frame..!!
    #so this dataframe_train basically contains 7 data colums, x1 through x6 and y
    dataframe = pd.read_csv(inputFileName).sample(frac=1, random_state=67).reset_index(drop=True)

    #iloc arguments => [row_positions, column_positions]
    #syntax is very similar to the how range(a:b:c) works..!
    raw_data = dataframe.iloc[:, 0:6].values       #take all rows but only 0-5 columns..!

    #now get the labels..!, i.e Y
    Y_raw = dataframe.iloc[:, 6:7].values          #a numpy nd array..!

    #N = no of samples, d = no of parameters..!
    N_total, _ = raw_data.shape

    X_expanded, feature_degrees = build_polynomial_design_matrix(raw_data, max_degree=10)

    #Train / validation split..!
    split_idx = int(0.8 * N_total)

    #make the data matrix split
    X_train = X_expanded[:split_idx]
    X_val = X_expanded[split_idx:]

    #make the label split
    Y_train = Y_raw[:split_idx]
    Y_val = Y_raw[split_idx:]

    #Define some lamda values..! and search among them..!

    #initially, [0.0001, 0.001, 0.01, 0.1, 1, 10] were used to find the approximate order of lamda needed
    #[0.01, ] had the best result among those, so values around 0.01 were chosen, for the next search..!

    #next, [0.005, 0.01, 0.03, 0.05] are chosen and [0.005, ] had the best results..! so, next search around that..!
    #next, [0.003, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007] are chosen..!, but the best is still around 0.005..!
    lambda_values = [0.002, 0.003, 0.004, 0.0045, 0.005, 0.0055, 0.006, 0.007, 0.008]

    best_lambda = None
    #apprarently this is how you denote -inf in python! :O
    best_val_r2 = -float('inf')
    best_weights = None

    if debug:
        print(f"{'Lambda':<10} | {'Val R^2':<12} | {'Val MSE':<12} | {'Features Kept'}")
        print("-" * 65)

    #for each lamda in the set, minimize the loss function using gradient descent..!
    for current_lambda in lambda_values:
        #treating this as a black box, there is L1 regularizatino running underneath.
        #not too sure of the syntax..!
        lasso = Lasso(alpha=current_lambda, fit_intercept=True, max_iter=100000)
        lasso.fit(X_train[:, 1:], Y_train.ravel())

        w = np.concatenate((np.array([lasso.intercept_]), lasso.coef_)).reshape(-1, 1)

        #evaluate on the validation set
        Y_val_hat = X_val @ w
        val_r2 = r_squared(Y_val, Y_val_hat)
        val_mse = mean_squared_error(Y_val, Y_val_hat)

        #syntactic sugar for knowing which of the weights are not zero..!
        features_kept = np.sum(w != 0)

        if debug:
            print(f"{current_lambda:<10} | {val_r2:<12.4f} | {val_mse:<12.4f} | {features_kept}")

        #keep track of the best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_lambda = current_lambda
            best_weights = np.copy(w)

    if debug:
        print("-" * 65)
        print(f"Optimal Lambda: {best_lambda} (Validation R^2: {best_val_r2:.4f})")

    assert best_weights is not None

    return best_lambda, best_weights, feature_degrees

#for this is have taken help from AI, cuz i dont know a lot of technicality regarding
#numpy or a lot of ways to work with numpy..!
def summarize_feature_degrees(best_weights, feature_degrees):
    non_zero_mask = np.abs(best_weights.ravel()) > 1e-12
    active_degrees = np.array(feature_degrees)[non_zero_mask]

    unique_degrees, counts = np.unique(active_degrees, return_counts=True)

    print(f"\n{'degree':<20} {'no of features'}")
    print("-" * 34)
    for deg, count in sorted(zip(unique_degrees, counts), reverse=True):
        print(f"{deg:<20} {count}")

    return unique_degrees, counts


def make_final_prediction(best_weights, inputFileName, outputFileName):
    # finally make the prediction based on the best model made so far..!
    df_test = pd.read_csv(inputFileName)
    raw_test_data = df_test.iloc[:, 0:6].values
    N_test = raw_test_data.shape[0]

    #make the data matrix for the test data, using the same logic as earlier..!
    X_test_expanded = np.ones((N_test, 1))
    for degree in range(1, 11):
        combinations = list(combinations_with_replacement(list(range(6)), degree))
        for combination in combinations:
            selected_columns = raw_test_data[:, combination]
            next_column = np.prod(selected_columns, axis=1).reshape(-1, 1)
            X_test_expanded = np.hstack((X_test_expanded, next_column))

    #make the predictions
    Y_test_predictions = X_test_expanded @ best_weights

    #write to csv :
    submission_df = pd.DataFrame(Y_test_predictions, columns=['y'])
    submission_df.to_csv(outputFileName, index=False)
    print("Saved predictions to IMT2024003-pred_var1.csv")

    return Y_test_predictions


def main():
    best_lambda, best_weights, feature_degrees = find_best_weights("IMT2024003_train_var1.csv", debug=True)
    summarize_feature_degrees(best_weights, feature_degrees)
    make_final_prediction(best_weights, "IMT2024003_test_var1.csv", "IMT2024003_pred_var1.csv")

if __name__ == "__main__":
    main()