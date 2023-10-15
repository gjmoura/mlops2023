import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import style


def predict_stock_marker() -> None:
    """
        Predict stock marker
    """
    df_sphist = pd.read_csv("sphist.csv", parse_dates=["Date"])

    def prepare_data_frame() -> None:
        """
        Prepare dataframe for training
        """

        logging.info("Preparing DataFrame...")

        df_sphist.sort_values("Date", ignore_index=True, inplace=True)

        df_sphist["prev5_avg"] = 0
        df_sphist["prev5_std"] = 0
        df_sphist["prev30_avg"] = 0
        df_sphist["prev30_std"] = 0
        df_sphist["prev365_avg"] = 0
        df_sphist["prev365_std"] = 0

        for index in df_sphist.iterrows():
            past5 = df_sphist.iloc[index-5:index]
            past5_avg = past5["Close"].mean()
            past5_std = past5["Close"].std()
            df_sphist.loc[index, "prev5_avg"] = past5_avg
            df_sphist.loc[index, "prev5_std"] = past5_std

            past30 = df_sphist.iloc[index-30:index]
            past30_avg = past30["Close"].mean()
            past30_std = past30["Close"].std()
            df_sphist.loc[index, "prev30_avg"] = past30_avg
            df_sphist.loc[index, "prev30_std"] = past30_std

            past365 = df_sphist.iloc[index-365:index]
            past365_avg = past365["Close"].mean()
            past365_std = past365["Close"].std()
            df_sphist.loc[index, "prev365_avg"] = past365_avg
            df_sphist.loc[index, "prev365_std"] = past365_std

    prepare_data_frame()

    def plot_graph(test: pd.DataFrame, predictions: np.ndarray) -> None:
        """
        Plot the model vs actual values

        test: DataFrame with the data tests
        predictions: List with de predictions
        """
        style.use("fivethirtyeight")
        plt.figure(figsize=(15, 10))
        plt.plot(test["Date"], test["Close"])
        plt.plot(test["Date"], predictions)
        plt.legend(["Actual", "Predicted"])
        plt.show()

    def linear_regression_training() -> None:
        """
            Linear Regression model with the 6 possible predictors
        """
        sliced_df_sphist = df_sphist.iloc[365:]

        train_df = sliced_df_sphist[sliced_df_sphist["Date"]
                                    < datetime(year=2013, month=1, day=1)]
        test_df = sliced_df_sphist[sliced_df_sphist["Date"]
                                   >= datetime(year=2013, month=1, day=1)]

        linear_regression = LinearRegression()
        linear_regression.fit(train_df[["prev5_avg", "prev5_std", "prev30_avg",
                              "prev30_std", "prev365_avg", "prev365_std"]], train_df["Close"])
        predictions_list = linear_regression.predict(
            test_df[["prev5_avg", "prev5_std", "prev30_avg", "prev30_std", "prev365_avg", "prev365_std"]])
        rmse = mean_squared_error(predictions_list, test_df["Close"]) ** 1/2

        plot_graph(test_df, predictions_list)

        print("RMSE value:", round(rmse, 1))

    linear_regression_training()


predict_stock_marker()
