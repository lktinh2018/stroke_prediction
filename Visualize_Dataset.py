import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from global_vars import *

df = pd.read_csv(INPUT_CSV_PATH)

for col_name in df.columns:
    if col_name != GT_COL_NAME:
        if col_name == "gender":

            def my_fmt(x):
                return "{:.2f}%\n({:.0f})".format(x, total * x / 100)

            gender_values = df[col_name].to_list()
            stroke_values = df[GT_COL_NAME].to_list()

            total = len(gender_values)
            c = Counter(gender_values)
            labels = list(c.keys())
            plt.pie(list(c.values()), labels=labels, autopct=my_fmt)
            plt.legend()
            plt.show()

            # Get genders is stroked
            stroke_by_gender_values = []
            for gender_value, stroke_value in zip(gender_values, stroke_values):
                if stroke_value == 1:
                    stroke_by_gender_values.append(gender_value)

            total = len(stroke_by_gender_values)
            c = Counter(stroke_by_gender_values)
            labels = list(c.keys())
            plt.pie(list(c.values()), labels=labels, autopct=my_fmt)
            plt.legend()
            plt.show()

    # if col_name == "stroke":

    #     def my_fmt(x):
    #         return "{:.2f}%\n({:.0f})".format(x, total * x / 100)

    #     gender_values = df[col_name].to_list()
    #     stroke_values = df[GT_COL_NAME].to_list()

    #     total = len(gender_values)
    #     c = Counter(gender_values)
    #     labels = list(c.keys())
    #     plt.pie(list(c.values()), labels=labels, autopct=my_fmt)
    #     plt.legend()
    #     plt.show()
