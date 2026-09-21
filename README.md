# Oil Pipeline Pressure Monitoring & Leak Detection System

A complete **working software prototype** for an industrial Oil Pipeline Pressure Monitoring and Leak Detection System.

This project uses synthetic multi-sensor pipeline data, feature engineering, line-pack mass balance correction, mode-specific Machine Learning (Isolation Forest), Negative Pressure Wave (NPW) wave front timing, and multi-evidence fusion to detect, classify, locate, and alert on pipeline leaks.

---

## Architecture

```text
Pressure Sensors (10 Hz) + Flow Sensors (1 Hz)
                    ↓
           Data Preprocessing
                    ↓
           Feature Engineering
                    ↓
            Operating Mode
       ┌────────────┴────────────┐
       ↓                         ↓
Isolation Forest          Flow Imbalance
per Operating Mode     + Line-Pack Correction
       │                         │
       └────────────┬────────────┘
                    ↓
    Negative Pressure Wave Detector
                    ↓
             Evidence Fusion
                    ↓
            Confidence Score
                    ↓
           Leak Classification
                    ↓
            Leak Localization
                    ↓
            Control Room Alert
```

---

## Objective

The system monitors joint behavior between multi-station pressure streams (10 Hz) and flow rate sensors (1 Hz) along a 100 km oil pipeline to:
1. Detect small chronic leaks that stay beneath simple static thresholds.
2. Prevent false alarms caused by normal operational transitions (e.g. pump ramping).
3. Combine 3 independent evidence streams into a single confidence score.
4. Estimate exact physical leak location (in km) using acoustic wave timing and hydraulic drop profiles.
5. Provide clear recommended actions for control room operators.

---

## Technologies Used

- **Python 3**: Core language.
- **NumPy & Pandas**: Signal preprocessing, rolling window calculations, feature extraction.
- **SciPy**: Numerical gradient calculations and acoustic wave equations.
- **Scikit-Learn (Isolation Forest)**: Operating-mode specific anomaly detection.
- **Joblib**: Model serialization (.joblib).
- **Matplotlib & Plotly**: Static report generation and interactive UI visualizations.
- **Streamlit**: Interactive web dashboard UI.

### Note on ZEDEDA
> **ZEDEDA** represents a **future edge deployment and orchestration layer** for distributing trained models to remote pipeline edge nodes (e.g., pump station edge gateways). **ZEDEDA is NOT a dependency for this local prototype** and is not required to run the code.

---

## Installation

```bash
# 1. Clone or navigate into the project directory
cd Oil-Pipeline-Monitoring-System

# 2. Create a Python virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 4. Install required dependencies
pip install -r requirements.txt
```

---

## How to Run

### Step 1: Run the Full Execution Pipeline
Executes synthetic data generation, feature engineering, model training, NPW wave detection, evidence fusion, leak localization, static plot creation, and demo report summary generation:

```bash
python run.py
```

Outputs created:
- `data/raw/synthetic_pipeline_data.csv`: Generated multi-station sensor streams.
- `data/processed/pipeline_features.csv`: Engineered diagnostic features.
- `data/processed/detection_results.csv`: Complete detection & localization results.
- `models/isolation_forest_*.joblib`: Saved mode-specific ML models.
- `reports/*.png`: 7 static validation plots.
- `reports/demo_results.txt`: Summary demo results text report.

### Step 2: Run the Interactive Control Room Dashboard
Launches the Streamlit web interface:

```bash
streamlit run dashboard/app.py
```

Then open your browser to `http://localhost:8501`.

---

## How the AI & Detection Algorithms Work

### 1. Line-Pack Mass Balance Correction
In real oil pipelines, oil is slightly compressible under high pressure. When a pump ramps up, oil accumulates temporarily inside the pipe before exiting, causing inlet flow to exceed outlet flow naturally (*line-pack charging*).
Our prototype uses a simplified rate-of-change correction:
$$\text{Corrected Imbalance} = (Q_{in} - Q_{out}) - K_{lp} \cdot \frac{d\bar{P}}{dt}$$
This prevents normal pump ramps from triggering false leak alarms.

### 2. Operating Mode Specific Isolation Forest
Normal pipeline behavior varies dramatically depending on whether the pipeline is steady **Flowing**, **Ramping**, or **Shut-in**.
Instead of one generic ML model, our system trains **separate Isolation Forest models** for each operating mode on clean baseline data. Isolation Forest works by randomly isolating data points: normal samples require many splits to isolate, while anomalous samples (leaks) isolate quickly, yielding higher anomaly scores.

### 3. Negative Pressure Wave (NPW) Timing & Acoustic Localization
When a large leak or pipe wall rupture occurs, a sharp negative pressure wave front travels outward in both directions at the speed of sound in oil ($v \approx 1000 \text{ m/s} = 1.0 \text{ km/s}$).
High-frequency 10 Hz pressure sensors detect the exact wave arrival timestamps ($t_A$ and $t_B$) at surrounding stations ($x_A$ and $x_B$).
The acoustic wave location formula pinpoints the exact rupture location:
$$x_{leak} = \frac{(x_A + x_B) - v_{wave} \cdot (t_B - t_A)}{2}$$

### 4. Multi-Evidence Fusion & Persistence Filter
Three signals are normalized into $[0, 1]$ evidence scores and fused via weighted combination ($40\%\text{ ML} + 30\%\text{ Flow} + 30\%\text{ NPW}$):
$$\text{Fused Score} = 0.40 \cdot S_{ML} + 0.30 \cdot S_{Flow} + 0.30 \cdot S_{NPW}$$
A 4-second rolling persistence filter ensures transient electrical noise or quick valve taps do not trigger false alerts.

---

## System Limitations

- **Synthetic Data**: Prototype operates on physics-based synthetic sensor data.
- **Simplified Hydraulic Model**: Uses linear approximations for pressure drops rather than non-linear Navier-Stokes/transient fluid flow equations.
- **Simplified Line-Pack Storage**: Uses a constant coefficient $K_{lp}$ rather than real-time fluid density, compressibility, and pipe thermal expansion models.
- **No Cloud/Hardware Dependency**: Designed as a standalone local desktop prototype for project evaluation.

---

## Future Work

- Integration with real industrial IoT pressure sensors and PLC telemetry streams.
- Full transient fluid hydraulic simulation (e.g. Synergi Pipeline Simulator / Stoner Pipeline Simulator integration).
- **ZEDEDA edge node orchestration** for deploying trained model containers to field pump gateways.
- Real-time Kafka / MQTT streaming data pipeline.
