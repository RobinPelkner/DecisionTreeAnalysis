import pandas as pd

data = pd.read_csv("data/heart_2020_cleaned.csv", sep=",")
data.dropna()
data["Diabetic"] = data["Diabetic"].replace(
    {"No": 0, "No, borderline diabetes": 1, "Yes (during pregnancy)": 2, "Yes": 3}
)

data["Sex"] = data["Sex"].replace({"Female": 0, "Male": 1})

data["Race"] = data["Race"].replace(
    {
        "American Indian/Alaskan Native": 0,
        "Asian": 1,
        "Black": 2,
        "Hispanic": 3,
        "Other": 4,
        "White": 5,
    }
)

data["AgeCategory"] = data["AgeCategory"].replace(
    {
        "18-24": 0,
        "25-29": 1,
        "30-34": 2,
        "35-39": 3,
        "40-44": 4,
        "45-49": 5,
        "50-54": 6,
        "55-59": 7,
        "60-64": 8,
        "65-69": 9,
        "70-74": 10,
        "75-79": 11,
        "80 or older": 12,
    }
)

data["GenHealth"] = data["GenHealth"].replace(
    {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
)

data = data.replace({"No": 0, "Yes": 1})

print(data.head())

data.to_csv("data/heart_prep.csv", index=False)