# AISI 1045 Electrical Resistance Dataset and Jupyter Analysis Examples

This repository contains experimental CSV data and Jupyter Python notebooks for exploring electrical resistance estimation in AISI 1045 carbon steel samples.

The main objective of this repository is to provide an open dataset that can be used to test, compare, and develop different data-analysis methods for electrical resistance estimation, signal preprocessing, parameter estimation, and related exploratory studies.

The included Least-Squares estimator is provided as a reference example. Users are encouraged to implement and compare additional estimators or signal-processing approaches.

---

## 1. Repository purpose

This repository is designed for:

- Interactive analysis using Jupyter notebooks.
- Reading and organizing CSV files by sample and experiment.
- Applying basic signal correction and scaling.
- Testing resistance estimators.
- Developing new methods for data analysis.
- Comparing estimators using the same raw experimental data.

The dataset comes from experiments where voltage, current, and temperature signals were acquired from AISI 1045 carbon steel samples using a controlled current excitation. The experimental work considers 10 steel samples and 10 experiments per sample, giving a total of 100 CSV files. The CSV files were extracted directly from the oscilloscope.

---

## 2. Recommended repository structure

The suggested repository organization is:

```text
.
├── data/
│   ├── sample0/
│   │   ├── experiment0.csv
│   │   ├── experiment1.csv
│   │   ├── experiment2.csv
|   |   ├── ...
│   │   └── experiment9.csv
│   ├── sample1/
│   │   ├── experiment0.csv
│   │   └── ...
│   └── sample9/
│       ├── experiment0.csv
│       └── experiment9.csv
├── notebooks/
│   └── main.ipynb
│
├── results/
│   └── resistance_results.csv
│
├── requirements.txt
└── README.md
```

The `data/` directory contains the experimental CSV files. Each sample folder contains the experiments associated with that sample.

The notebook assumes the following naming pattern:

```text
sample*
experiment*.csv
```

Examples:

```text
data/sample0/experiment0.csv
data/sample0/experiment1.csv
data/sample1/experiment0.csv
```

---

## 3. Dataset organization

The data are organized by sample and experiment.

Each folder named `sample0`, `sample1`, ..., `sample9` corresponds to one steel sample. Each CSV file named `experiment0.csv`, `experiment1.csv`, ..., `experiment9.csv` corresponds to one repeated experiment for that sample.

The complete dataset contains:

```text
10 samples × 10 experiments = 100 CSV files
```

---

## 4. Expected CSV format

The CSV files are imported using a predefined list of column names because the oscilloscope export includes several unused columns.

The notebook assigns the following column names during import:

```python
names=[
    "Unnamed 1", "Info", "Unnamed 2", "Time", "Current",
    "Unnamed 3", "Unnamed 4", "Unnamed 5", "Unnamed 6",
    "Time 2", "Temperature", "Unnamed 7", "Unnamed 8",
    "Unnamed 9", "Unnamed 10", "Time 3", "Voltage",
    "Unnamed 11"
]
```

After importing the CSV file, only the following columns are retained:

| Column        | Description                             |
| ------------- | --------------------------------------- |
| `Time`        | Time vector used for the current signal |
| `Current`     | Measured current signal                 |
| `Temperature` | Measured temperature signal             |
| `Voltage`     | Measured voltage signal                 |

The resulting dataframe contains:

```python
df[["Time", "Current", "Temperature", "Voltage"]]
```

---

## 5. Running the notebook

Open the notebook with Jupyter or use visual studio code with Jupyter as Extension. Note: if your prefer you can use plain Python and use the code to obtain the same results:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

Then open:

```text
notebooks/main.ipynb
```

The notebook is designed to be run cell by cell.

The current code assumes that the notebook is located inside the `notebooks/` folder and that the `data/` folder is one level above it. Therefore, the dataset is accessed using:

```python
"../data"
```

For example:

```python
results = ProcessDataset("../data")
```

If the notebook is moved to the repository root, change the path to:

```python
results = ProcessDataset("data")
```

---

# Notebook cell guide

This section describes the purpose of each cell in `main.ipynb`. The notebook is intended for interactive use with Jupyter and provides a basic workflow for reading CSV files, scaling signals, and testing resistance estimators. **The code in each cell is compatible with Pytho 3 and can be used as the base of your own code to run over other Python-based IDEs.**

---

## Cell 1 — Import libraries and define constants

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PHASE_SHIFT_SAMPLES = 0  # each count = 20e-6 s
VOLTAGE_GAIN = 4.0       # voltage scaling factor
TEMPERATURE_SCALE = 0.01 # temperature scaling factor
ts = 20E-6               # sampling period: 20 us
```

This cell imports the required Python libraries and defines the constants used during preprocessing.

| Library             | Purpose                         |
| ------------------- | ------------------------------- |
| `numpy`             | Numerical operations            |
| `pandas`            | Reading and organizing CSV data |
| `matplotlib.pyplot` | Plotting signals                |
| `pathlib.Path`      | Managing file paths             |

| Constant              | Meaning                                            |
| --------------------- | -------------------------------------------------- |
| `PHASE_SHIFT_SAMPLES` | Number of samples used to shift the voltage signal |
| `VOLTAGE_GAIN`        | Voltage scaling factor                             |
| `TEMPERATURE_SCALE`   | Temperature scaling factor                         |
| `ts`                  | Sampling period                                    |

Each count in `PHASE_SHIFT_SAMPLES` corresponds to one sampling period:

```text
1 count = 20 us
```

For example:

```python
PHASE_SHIFT_SAMPLES = 160
```

corresponds to:

```text
160 × 20 us = 3.2 ms
```

If `PHASE_SHIFT_SAMPLES = 0`, no phase-shift correction is applied.

---

## Cell 2 — Import and preprocess one experiment

```python
def ReadExperiment(csvFile):
    dfTemp = pd.read_csv(
        csvFile,
        sep=",",
        names=[
            "Unnamed 1", "Info", "Unnamed 2", "Time", "Current",
            "Unnamed 3", "Unnamed 4", "Unnamed 5", "Unnamed 6",
            "Time 2", "Temperature", "Unnamed 7", "Unnamed 8",
            "Unnamed 9", "Unnamed 10", "Time 3", "Voltage",
            "Unnamed 11"
        ],
        low_memory=False
    )

    df = dfTemp[["Time", "Current", "Temperature", "Voltage"]].copy()

    df["Voltage"] = np.roll(df["Voltage"].to_numpy(), -PHASE_SHIFT_SAMPLES)
    df["Voltage"] = df["Voltage"] / VOLTAGE_GAIN
    df["Temperature"] = df["Temperature"] / TEMPERATURE_SCALE

    return df
```

This cell defines the function:

```python
ReadExperiment(csvFile)
```

The function reads one CSV file and returns a dataframe containing only the relevant variables: Time, Current, Temperature, Voltage.

The function performs four operations:

1. Reads the CSV file.
2. Selects the relevant columns.
3. Applies phase-shift correction to the voltage signal.
4. Scales the voltage and temperature signals.

The voltage signal is shifted using:

```python
np.roll(df["Voltage"].to_numpy(), -PHASE_SHIFT_SAMPLES)
```

The negative sign shifts the voltage signal to the left. If you preffer to move the signal in the oposite direction, change `-PHASE_SHIFT_SAMPLES` to `PHASE_SHIFT_SAMPLES`.

The voltage signal is scaled with:

```python
df["Voltage"] = df["Voltage"] / VOLTAGE_GAIN
```

The temperature signal is scaled with:

```python
df["Temperature"] = df["Temperature"] / TEMPERATURE_SCALE
```

---

## Cell 3 — Read one experiment example

```python
exp = "../data/sample0/experiment0.csv"
data = ReadExperiment(exp)
data
```

This cell reads one experiment manually.

The selected file is:

```text
../data/sample0/experiment0.csv
```

This step is useful for checking whether a single CSV file can be imported correctly before processing the complete dataset.

Users can change the file path to inspect another experiment:

```python
exp = "../data/sample3/experiment5.csv"
data = ReadExperiment(exp)
data
```

---

## Cell 4 — Plot one signal

To inspect the previously imported data use:

```python
time = data["Time"]
voltage = data["Voltage"]
current = data["Current"]

plt.plot(time, voltage, ".k", label="voltage")
plt.legend()
plt.show()
```

This cell plots the voltage signal from the selected experiment.

It is useful for visually checking:

- Signal shape.
- Signal amplitude.
- Noise level.
- Possible import errors.
- Whether scaling has been applied correctly.

Users may also plot current and voltage together:

```python
plt.plot(time, current, label="current")
plt.plot(time, voltage, label="voltage")
plt.legend()
plt.show()
```

Because current and voltage may have different magnitudes, separate plots may be more appropriate for detailed inspection.

---

## Cell 5 — Define the example estimator

```python
def EstimateRes(voltage, current):
    # example of least-squares estimator
    # try your own estimator
    return np.sum(voltage * current) / np.sum(current * current)
```

This cell defines the function:

```python
EstimateRes(voltage, current)
```

This is an example Least-Squares resistance estimator. It assumes the model:

```text
u(t) = R i(t)
```

where:

| Symbol | Meaning               |
| ------ | --------------------- |
| `u(t)` | Voltage               |
| `i(t)` | Current               |
| `R`    | Electrical resistance |

The implemented estimator is:

```text
R = sum(u(t) i(t)) / sum(i(t)^2)
```

## This estimator is included only as a reference example.

## Cell 6 — Process all samples and experiments

```python
def ProcessDataset(dataDir):
    dataDir = Path(dataDir)
    results = []

    for sample_dir in sorted(dataDir.glob("sample*")):
        for csv_file in sorted(sample_dir.glob("experiment*.csv")):
            df = ReadExperiment(csv_file)

            resistance = EstimateRes(
                df["Voltage"].to_numpy(),
                df["Current"].to_numpy()
            )

            results.append({
                "sample": sample_dir.name,
                "experiment": csv_file.stem,
                "resistance_ohm": resistance,
                "temperature_mean_celsius": df["Temperature"].mean()
            })

    return pd.DataFrame(results)
```

This cell defines the function:

```python
ProcessDataset(dataDir)
```

The function processes all experiments in the dataset.

It searches for sample folders using:

```python
dataDir.glob("sample*")
```

and searches for CSV files using:

```python
sample_dir.glob("experiment*.csv")
```

Therefore, the expected naming format is:

```text
sample0, sample1, sample2, ...
experiment0.csv, experiment1.csv, experiment2.csv, ...
```

The function returns a dataframe with one row per experiment.

| Output column              | Description                         |
| -------------------------- | ----------------------------------- |
| `sample`                   | Sample folder name                  |
| `experiment`               | Experiment file name without `.csv` |
| `resistance_ohm`           | Estimated electrical resistance     |
| `temperature_mean_celsius` | Mean temperature for the experiment |

---

## Cell 7 — Run the full dataset processing

```python
results = ProcessDataset("../data")
results
```

This cell applies the full processing workflow to all samples and experiments located in:

```text
../data
```

If the dataset contains 10 samples and 10 experiments per sample, the expected number of rows is:

```text
100
```

Users can verify this with:

```python
len(results)
```

If the notebook is located in the repository root instead of the `notebooks/` folder, use:

```python
results = ProcessDataset("data")
```

---

# Requirements

The notebook requires Python 3 and the following packages:

```text
numpy
pandas
matplotlib
jupyter
```

A minimal `requirements.txt` file is:

```text
numpy
pandas
matplotlib
jupyter
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

# Reproducibility notes

To preserve reproducibility:

- Keep raw CSV files unchanged.
- Use notebooks or scripts to generate processed data.
- Save generated results in a separate `results/` folder.
- Document any change in scaling factors.
- Document the selected value of `PHASE_SHIFT_SAMPLES`.
- Report whether the phase correction is enabled or disabled.
- Indicate the estimator used to generate each result table.

---

# Data availability statement

The CSV files in this repository are provided as experimental data for testing electrical resistance estimation methods in AISI 1045 carbon steel samples.

The dataset is organized by sample and experiment. The included Jupyter notebook provides a basic import, preprocessing, and estimation workflow. The Least-Squares estimator is included only as an example, and the repository is intended to support the development and comparison of new signal-processing and data-analysis methods.

---

# Suggested citation

If this dataset or notebook is used, please cite the associated manuscript:

> Electrical Resistance-Based Characterization of AISI 1045 Carbon Steel Using Controlled Current Injection and Parameter Estimation.

**Note: there is missing information about the manuscript, this should be updated after its publication.**
