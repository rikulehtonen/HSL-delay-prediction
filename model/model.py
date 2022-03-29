import json
import numpy as np                   # import numpy package under shorthand "np"
import pandas as pd                  # import pandas package under shorthand "pd"
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures    # function to generate polynomial and interaction features
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error
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
    ax.plot(t,y_pred, color="black")

    X_val = best_hub_model[0].fit_transform(t)
    y_pred = best_hub_model[1].predict(X_val)
    ax.plot(t,y_pred, color="red")

    plt.ylim([-300, 1300])
    plt.title(title)
    plt.legend(["Data point","Linear regr","Huber regr"])
    plt.xlabel("Scheduled Arrival Time (s)")
    plt.ylabel("Arrival Delay (s)")
    plt.show()

def model(df,stop,line,days,title):
    newData= df[(df['stopName'] == stop) & (df['routeShortName'] == line) & (df['serviceDay'].isin(days))]
    X = newData['scheduledArrival'].to_numpy().reshape(-1, 1)
    y = newData['arrivalDelay'].to_numpy()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    validation_lin_errors = []
    validation_hub_errors = []
    models_lin = []
    models_hub = []
    degrees = range(1,5)
    for i in range(len(degrees)):
        degree = degrees[i]
        validation_lin_errors.append([])
        validation_hub_errors.append([])
        models_lin.append([])
        models_hub.append([])
        for j, (train_indices, val_indices) in enumerate(cv.split(X)):
            
            X_train, y_train, X_val, y_val = X[train_indices], y[train_indices], X[val_indices], y[val_indices]
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train)
            X_val_poly = poly.fit_transform(X_val)

            #### LinearRegression
            lin_regr = LinearRegression(fit_intercept=False)
            lin_regr.fit(X_train_poly, y_train)
            y_pred_lin_val = lin_regr.predict(X_val_poly)
            val_lin_error = mean_squared_error(y_val, y_pred_lin_val)

            #### HuberRegressor
            hub_regr = HuberRegressor(fit_intercept=False, epsilon=1.0)
            hub_regr.fit(X_train_poly, y_train)
            y_pred_hub_val = hub_regr.predict(X_val_poly)
            val_hub_error = mean_squared_error(y_val, y_pred_hub_val)

            models_lin[i].append((poly,lin_regr))
            models_hub[i].append((poly,hub_regr))
            validation_lin_errors[i].append(val_lin_error)
            validation_hub_errors[i].append(val_hub_error)

    #### lin_regr select best degree and model:
    average_lin_error = [sum(err) / len(err) for err in validation_lin_errors]
    best_lin_degree = average_lin_error.index(min(average_lin_error))
    best_lin_model = models_lin[best_lin_degree][validation_lin_errors[best_lin_degree].index(min(validation_lin_errors[best_lin_degree]))]

    #### hub_regr select best degree and model:
    average_hub_error = [sum(err) / len(err) for err in validation_hub_errors]
    best_hub_degree = average_hub_error.index(min(average_hub_error))
    best_hub_model = models_lin[best_hub_degree][validation_hub_errors[best_hub_degree].index(min(validation_hub_errors[best_hub_degree]))]

    #### Create plot
    plot((X,y), best_lin_model, best_hub_model, range(20000,86400), title)





def createModels(jsond):
    file = jsond.get("data").get("files")[0]
    df = pd.read_csv(file)
    daysets = jsond.get("model").get("daysets")
    stops = jsond.get("model").get("stops")
    lines = jsond.get("model").get("lines")
    titles = jsond.get("model").get("setNames")

    for i in range(len(stops)):
        for n in range(len(daysets)):
            model(df,stops[i],lines[i],daysets[n],titles[n])

def main():
    fname = "dataread.json"
    jsond = loadData(fname)
    createModels(jsond)


if __name__ == "__main__":
    main()