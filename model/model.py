import json
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold


def loadData(fname):
    with open(fname) as f:
        jsond = json.load(f)
        return jsond

def plot(points, best_lin_model, best_hub_model, timeRange, title):
    ## arrival range (time of the day in seconds)
    t = np.array(timeRange).reshape(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(points[0],points[1])

    X_val = best_lin_model[0].fit_transform(t)
    y_pred = best_lin_model[1].predict(X_val)
    ax.plot(t,y_pred, color="black", linewidth=3)

    X_val = best_hub_model[0].fit_transform(t)
    y_pred = best_hub_model[1].predict(X_val)
    ax.plot(t,y_pred, color="orange", linewidth=3)

    plt.ylim([-300, 1300])
    plt.title(title)
    plt.legend(["Data point","Linear regr","Huber regr"])
    plt.xlabel("Scheduled Arrival Time (s)")
    plt.ylabel("Arrival Delay (s)")
    plt.show()

def plot_final(points, best_lin_model, best_hub_model, timeRange, title):
    ## arrival range (time of the day in seconds)
    t = np.array(timeRange).reshape(-1, 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(points[0],points[1])

    X_val = best_lin_model[0].fit_transform(t)
    y_pred = best_lin_model[1].predict(X_val)
    ax.plot(t,y_pred, color="black", linewidth=3)

    X_val = best_hub_model[0].fit_transform(t)
    y_pred = best_hub_model[1].predict(X_val)
    ax.plot(t,y_pred, color="orange", linewidth=3)

    plt.ylim([-300, 1300])
    plt.title(title)
    plt.legend(["Data point","Linear regr","Huber regr"])
    plt.xlabel("Scheduled Arrival Time (s)")
    plt.ylabel("Arrival Delay (s)")
    plt.show()


def model(df,stop,line,days,testdays,title, testTitle):
    newData= df[(df['stopName'] == stop) & (df['routeShortName'] == line) & (df['serviceDay'].isin(days))]
    #Training and validation data
    X = newData['scheduledArrival'].to_numpy().reshape(-1, 1)
    y = newData['arrivalDelay'].to_numpy()
    #Test data
    testData= df[(df['stopName'] == stop) & (df['routeShortName'] == line) & (df['serviceDay'].isin(testdays))]
    X_test = testData['scheduledArrival'].to_numpy().reshape(-1, 1)
    y_test = testData['arrivalDelay'].to_numpy()


    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validation_lin_errors = []
    validation_hub_errors = []
    train_lin_errors = []
    train_hub_errors = []
    test_lin_errors = []
    test_hub_errors = []
    models_lin = []
    models_hub = []
    degrees = range(1,5)
    for i in range(len(degrees)):
        degree = degrees[i]
        validation_lin_errors.append([])
        validation_hub_errors.append([])
        train_lin_errors.append([])
        train_hub_errors.append([])
        test_lin_errors.append([])
        test_hub_errors.append([])
        models_lin.append([])
        models_hub.append([])
        for j, (train_indices, val_indices) in enumerate(cv.split(X)):
            
            X_train, y_train, X_val, y_val = X[train_indices], y[train_indices], X[val_indices], y[val_indices]
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_val_poly = poly.fit_transform(X_val)
            X_test_poly = poly.fit_transform(X_test)

            #### LinearRegression
            lin_regr = LinearRegression(fit_intercept=False)
            lin_regr.fit(X_train_poly, y_train)
            y_pred_lin_train = lin_regr.predict(X_train_poly)
            y_pred_lin_val = lin_regr.predict(X_val_poly)
            y_pred_lin_test = lin_regr.predict(X_test_poly)
            train_lin_errors[i].append(mean_squared_error(y_train, y_pred_lin_train))
            validation_lin_errors[i].append(mean_squared_error(y_val, y_pred_lin_val))
            test_lin_errors[i].append(mean_squared_error(y_test, y_pred_lin_test))

            #### HuberRegressor
            hub_regr = HuberRegressor(fit_intercept=False)
            hub_regr.fit(X_train_poly, y_train)
            y_pred_hub_train = hub_regr.predict(X_train_poly)
            y_pred_hub_val = hub_regr.predict(X_val_poly)
            y_pred_hub_test = hub_regr.predict(X_test_poly)
            train_hub_errors[i].append(mean_squared_error(y_train, y_pred_hub_train))
            validation_hub_errors[i].append(mean_squared_error(y_val, y_pred_hub_val))
            test_hub_errors[i].append(mean_squared_error(y_test, y_pred_hub_test))

            models_lin[i].append((poly,lin_regr))
            models_hub[i].append((poly,hub_regr))

    print("########################")
    print(title)

    #### lin_regr select best degree and model:
    average_lin_error = [sum(err) / len(err) for err in validation_lin_errors]
    average_lin_error_train = [sum(err) / len(err) for err in train_lin_errors]
    average_lin_error_test = [sum(err) / len(err) for err in test_lin_errors]
    best_lin_degree = average_lin_error.index(min(average_lin_error))
    best_lin_model_num = validation_lin_errors[best_lin_degree].index(min(validation_lin_errors[best_lin_degree]))
    best_lin_model = models_lin[best_lin_degree][best_lin_model_num]
    print()
    print("lin_regr errors")
    print(average_lin_error)
    print(average_lin_error_train)
    print("best deg: " + str(degrees[best_lin_degree]))
    print("Training error:" + str(train_lin_errors[best_lin_degree][best_lin_model_num]))
    print("Validation error:" + str(validation_lin_errors[best_lin_degree][best_lin_model_num]))
    print("Test error:" + str(test_lin_errors[best_lin_degree][best_lin_model_num]))

    #### hub_regr select best degree and model:
    average_hub_error = [sum(err) / len(err) for err in validation_hub_errors]
    average_hub_error_train = [sum(err) / len(err) for err in train_hub_errors]
    average_hub_error_test = [sum(err) / len(err) for err in test_hub_errors]
    best_hub_degree = average_hub_error.index(min(average_hub_error))
    best_hub_model_num = validation_hub_errors[best_hub_degree].index(min(validation_hub_errors[best_hub_degree]))
    best_hub_model = models_lin[best_hub_degree][best_hub_model_num]
    print()
    print("hub_regr errors")
    print(average_hub_error)
    print(average_hub_error_train)
    print("best deg: " + str(degrees[best_hub_degree]))
    print("Training error:" + str(train_hub_errors[best_hub_degree][best_hub_model_num]))
    print("Validation error:" + str(validation_lin_errors[best_hub_degree][best_hub_model_num]))
    print("Test error:" + str(test_lin_errors[best_hub_degree][best_hub_model_num]))

    #### Create plot
    plot((X,y), best_lin_model, best_hub_model, range(20000,86400), title)
    plot((X_test,y_test), best_lin_model, best_hub_model, range(20000,86400), testTitle)


def createModels(jsond):
    files = jsond.get("data").get("files")
    csvreads = [pd.read_csv(file) for file in files]
    df = pd.concat(csvreads)
    daysets = jsond.get("model").get("daysets")
    testsets = jsond.get("model").get("testsets")
    stops = jsond.get("model").get("stops")
    lines = jsond.get("model").get("lines")
    titles = jsond.get("model").get("setNames")
    testTitles = jsond.get("model").get("testNames")

    for i in range(len(stops)):
        for n in range(len(daysets)):
            model(df,stops[i],lines[i],daysets[n],testsets[n],titles[n],testTitles[n])

def main():
    fname = "dataread.json"
    jsond = loadData(fname)
    createModels(jsond)


if __name__ == "__main__":
    main()