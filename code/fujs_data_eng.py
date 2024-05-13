import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.preprocessing import StandardScaler
import numpy as np


DATA = Path(__file__).absolute().parents[1] / "data"

def clean_data(df: pd.DataFrame, filename: Path) -> pd.DataFrame:
    # Drop all rows older than 1.1.2022
    df = df[df["valid"] >= "2022-01-01"]
    print(df["valid"].min())

    # Drop Unnamed: 0, id columns
    df.drop(columns=["Unnamed: 0"], inplace=True)

    # If max date is older than 31.08.2023, delete file
    if df.empty or max(df["valid"]) < "2023-08-10":
        filename.unlink()
        return 
    
    # Drop all rows between 8 PM and 8 AM
    df["valid"] = pd.to_datetime(df["valid"])
    df = df[(df["valid"].dt.hour >= 8) & (df["valid"].dt.hour < 20)]

    return df


def load_and_clean(_is_cleaned: bool = True) -> pd.DataFrame:
    data_path = DATA / "ams-data"
    all_data = data_path.rglob("data_id-*.csv")

    files = {}

    for data_file in all_data:
        data = pd.read_csv(data_file)
        
        # Check if data is empty and delete file
        if data.empty:
            data_file.unlink()
            continue

        if not _is_cleaned:
            data = clean_data(data, data_file)

        if data is not None:
            files[data_file] = data

    return files


def encode_time(data: dict) -> pd.DataFrame:
    for file, df in data.items():
        # Convert "valid" to month quartals
        df["valid"] = pd.to_datetime(df["valid"])
        df["month"] = df["valid"].dt.month
        df["season"] = [month%12 // 3 + 1 for month in df["month"]]

        # Create day period column
        df["day_period"] = df["valid"].dt.hour // 3

        # Dummy encode day period
        df = pd.get_dummies(df, columns=["day_period", "season"])

        df.fillna(0, inplace=True)
        data[file] = df

    return data

def merge_data(df: dict) -> None:
    # Merge all dataframes
    all_data = pd.concat(df.values(), ignore_index=True)

    # Save data
    all_data.to_csv(str(DATA / "all_data.csv"), index=False)


def model(data: pd.DataFrame) -> None:
    # Split data
    X = data.drop(columns=["valid", "g_sunrad"], axis=1)
    y = data["g_sunrad"]

    X_train, X_test, y_train, y_test = X.iloc[:int(0.8*len(X))], X.iloc[int(0.8*len(X)):], y.iloc[:int(0.8*len(y))], y.iloc[int(0.8*len(y)):]

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model
    model = gbr()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(np.mean((y_pred - y_test)**2))


def save_data(data: pd.DataFrame) -> None:
    data.to_csv(str(DATA / "all_data.csv"), index=False)

def main():
    df = pd.read_csv(DATA / "all_data.csv")

    model(df)

if __name__ == "__main__":
    main()