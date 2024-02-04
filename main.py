
from _csv import reader
import pandas as pd
import pyarrow
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score


def plot_comparison_cat(cat1, cat2, bev_cat):
    for cat in bev_cat:
        # Consider only the current category
        df = sbxData[sbxData.Beverage_category == cat]
        # If no poitns are in this category, then don't plot anything
        if (len(df['Beverage_prep']) != 0):
            plt.scatter(df[cat1].tolist(), df[cat2].tolist(), label=cat)
    plt.xlabel(cat1)
    plt.ylabel(cat2)
    plt.legend()
    plt.show()


def plot_comparison_prep(prep1, prep2, bev_prep):
    for prep in bev_prep:
        # Consider only the current prep method
        df = sbxData[sbxData.Beverage_prep == prep]
        # Don't plot empty data points
        if (len(df['Beverage_prep']) != 0):
            plt.scatter(df[prep1].tolist(), df[prep2].tolist(), label=prep)
    plt.xlabel(prep1)
    plt.ylabel(prep2)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Initializing Data
    sbxData = pd.read_csv("/Users/ritenpatel/PycharmProjects/SBXDatathon/Starbucks Datathon File.csv")

    # Getting a list of the beverage category and preparation methods
    bev_cat = sbxData['Beverage_category'].drop_duplicates().tolist()
    bev_prep = sbxData['Beverage_prep'].drop_duplicates().tolist()
    print(" Beverage prep list: \n", bev_prep)
    print(" Columns: \n", sbxData.columns.tolist())

    # Drop the entries where Caffeine is variable
    sbxData['Caffeine (mg)'] = pd.to_numeric(sbxData['Caffeine (mg)'], errors="coerce")
    sbxData.dropna(subset=['Caffeine (mg)'], inplace=True)
    pd.set_option('display.max_columns', None)

    # Remove low calorie and low caffeine drinks
    sbxData = sbxData[sbxData['Calories'] > 99]
    sbxData = sbxData[sbxData['Caffeine (mg)'] > 50]

    # Nice relationship
    plot_comparison_cat(' Sugars (g)', 'Cholesterol (mg)', bev_cat)

    # Linear regression with cholesterol on x axis and sugars on y axis
    y = sbxData[' Sugars (g)']
    x = sbxData[['Cholesterol (mg)']]

    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    Y_pred = linear_regressor.predict(x)
    # R Squared value is 0.9857
    print(linear_regressor.score(x, y))
    # Line of best fit comes to Sugars = 0.974*Cholesterol - 1.886
    print("Coefficients: \n", linear_regressor.coef_)
    print("Intercept: \n", linear_regressor.intercept_)

    # Convert the strings with %'s to integers that we can graph
    sbxData[' Calcium (% DV) '] = sbxData[' Calcium (% DV) '].str.rstrip('%').astype(float) / 100.0
    sbxData['Iron (% DV) '] = sbxData['Iron (% DV) '].str.rstrip('%').astype(float) / 100.0
    sbxData = sbxData[sbxData[' Calcium (% DV) '] > 0]
    sbxData = sbxData[sbxData['Iron (% DV) '] > 0]

    plot_comparison_prep(' Calcium (% DV) ', 'Iron (% DV) ', bev_prep)

    # Assess relationship between Sugar and Carbs
    plot_comparison_cat(' Sugars (g)', ' Total Carbohydrates (g) ', bev_cat)
    y = sbxData[' Sugars (g)']
    x = sbxData[[' Total Carbohydrates (g) ']]

    linear_regressor = LinearRegression()
    linear_regressor.fit(x, y)
    Y_pred = linear_regressor.predict(x)
    # R Squared value is 0.699
    print(linear_regressor.score(x, y))
    # Line of best fit comes to Sugars = 0.207*Carbs+6.487
    print("Coefficients: \n", linear_regressor.coef_)
    print("Intercept: \n", linear_regressor.intercept_)

    # Possible classification problem:Wh
    plot_comparison_cat('Calories', " Total Carbohydrates (g) ", bev_cat)

    # Remove the bev_prep = short, tall, grande, or venti
    filterList = ['Short', 'Tall', 'Grande', 'Venti']
    for x in filterList:
        filter = sbxData['Beverage_prep'].str.contains(x)
        sbxData = sbxData[~filter]

    # Calories and sodium tells us what kind of milk was used !!!!!
    plot_comparison_prep("Calories", " Sodium (mg)", bev_prep)

    # Let's run a KNN algorithm to predict the milk based off calories and sodium
    # Map the milk(categorical variable) to numbers
    sbxData['Beverage_prep'] = sbxData['Beverage_prep'].map({'2% Milk': 0, 'Soymilk': 1, 'Whole Milk': 2}).astype(int)

    # Remove everything from the data except the calories and sodium
    calories_list = sbxData['Calories'].tolist()
    sodium_list = sbxData[' Sodium (mg)'].tolist()
    data = [None] * len(calories_list)
    for i in range(len(calories_list)):
        data[i] = [calories_list[i], sodium_list[i]]

    # X data is the sodium and calories, Y data is the beverage prep method
    x_data = pd.DataFrame(data, columns=['Calories', ' Sodium (mg)'])
    y_data = sbxData['Beverage_prep']
    # Normalizing the data to values between 0 and 1 so we can compare them
    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(x_data)
    data = pd.DataFrame(X_data_minmax,
                        columns=['Calories', ' Sodium (mg)'])

    # Creating 80-20 train test split
    X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=0.2, random_state=1)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)

    ypred = knn_clf.predict(X_test)

    result1 = classification_report(y_test, ypred)
    print("Classification report: ", result1)

    result2 = accuracy_score(y_test, ypred)
    print("Accuracy: ", result2)
