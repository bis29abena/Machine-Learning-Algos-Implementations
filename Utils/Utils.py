import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Utils:
    def __init__(self, df):
        self.df = df

    def missing_values_table():
        """
            Prints out a dataframe with columns with missing values and their percentages
        """

        # Total missing values
        mis_val = self.df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * self.df.isnull().sum() / len(self.df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(self.df.shape[1]) + " columns.\n"
              "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns

    def plot_corr(size=11):
        """
            Functions plot a graphical correlation matrix for each pair of columns in the Dataframe

            Input:
                size: vertical and horizontal size of the plot

            Display:
                Matrix of correlation between columns. Blue-cyan-green-darkgreen-yellow => less to more correlated
                                                        0 ---------------------- 1
                                                        Expect a darkline runnning from top-left to bottom-right

        """
        corr = self.df.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), (corr.columns))
        plt.yticks(range(len(corr.columns)), (corr.columns))
