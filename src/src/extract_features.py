import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import os

def extract_features_from_csv(path):
    df = pd.read_csv(path)
    time = df["time"].values
    flux = df["flux"].values
    label = df["label"].values[0]

    # Basic stats
    mean_flux = np.mean(flux)
    std_flux = np.std(flux)
    amp = np.max(flux) - np.min(flux)
    percentile_95 = np.percentile(flux, 95)
    percentile_5 = np.percentile(flux, 5)

    # Lomb–Scargle: find period
    freq, power = LombScargle(time, flux).autopower()
    best_period = 1 / freq[np.argmax(power)]

    # Peak count
    peaks, _ = find_peaks(flux)
    num_peaks = len(peaks)

    features = {
        "mean_flux": mean_flux,
        "std_flux": std_flux,
        "amplitude": amp,
        "p95": percentile_95,
        "p5": percentile_5,
        "period": best_period,
        "num_peaks": num_peaks,
        "label": label
    }

    return features


def extract_all(data_dir="data", output_csv="features.csv"):
    rows = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            features = extract_features_from_csv(os.path.join(data_dir, file))
            rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")


if __name__ == "__main__":
    extract_all()
