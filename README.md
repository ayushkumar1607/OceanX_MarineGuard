# 🛢️ MarineGuard — From Slick Detection to Maritime Evidence

> **Smart India Hackathon 2026 | Problem Statement 26143 | NTRO**  
> *Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.*

**Team Ocean X**

---

## 🎯 Overview

MarineGuard is an end-to-end intelligent platform that investigates marine oil spills by combining satellite imagery (Sentinel-1 SAR), environmental data (wind & ocean currents), and historical AIS vessel traffic. The system detects oil slicks, traces them backward to their origin, and identifies the responsible vessel using a transparent, multi-factor attribution scoring system.

### The MarineGuard Principle
> *"Finds the vessel that best fits the incident, not just the closest."*

We never suspect a vessel based on proximity alone — our Explainable Attribution Score requires multiple corroborating factors for a high-confidence finding.

---

## 🏗️ Architecture

```
          Sentinel-1 SAR        ERA5 Wind + CMEMS Currents       Historical AIS Traffic
                │                         │                              │
                ▼                         │                              │
    ┌───────────────────┐                 │                              │
    │  Detection Engine │                 │                              │
    │  (U-Net + SAR)    │                 │                              │
    └────────┬──────────┘                 │                              │
             ▼                            ▼                              │
    ┌───────────────────┐    ┌────────────────────┐                      │
    │  Slick Properties │───▶│    Drift Engine     │                     │
    │  (loc, area, conf)│    │  (backward track)   │                     │
    └───────────────────┘    └────────┬───────────┘                      │
                                     ▼                                   │
                            ┌────────────────────┐                       │
                            │  Origin Zone +     │◀──────────────────────┘
                            │  Time Window       │
                            └────────┬───────────┘
                                     ▼
                            ┌────────────────────┐
                            │    AIS Engine       │
                            │  (filter + analyze) │
                            └────────┬───────────┘
                                     ▼
                            ┌────────────────────┐
                            │ Attribution Engine  │
                            │ (multi-factor score)│
                            └────────┬───────────┘
                                     ▼
                            ┌────────────────────┐
                            │  Investigation UI   │
                            │ (Map + Evidence)    │
                            └────────────────────┘
```

---

## 🔧 Pipeline

| Step | Engine | Description |
|------|--------|-------------|
| **Detect** | Detection Engine | U-Net segmentation on SAR imagery → slick location, area, confidence |
| **Trace** | Drift Engine | Backward Lagrangian particle tracking → probable origin zone + release window |
| **Correlate** | AIS Engine | Spatial-temporal vessel filtering + trajectory analysis + AIS gap detection |
| **Rank** | Attribution Engine | Multi-factor scoring: proximity (30%) + temporal (25%) + trajectory (20%) + AIS gap (15%) + behavioral (10%) |
| **Explain** | Investigation UI | Interactive map, ranked vessels, forensic timeline, "Why Ranked?" evidence panel |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Test Data

```bash
python data/generate_test_data.py
```

This generates 3 test scenarios:
- **Major Spill** — Arabian Sea, 15 km², clear single-vessel attribution
- **No Spill** — Bay of Bengal, false positive (SAR look-alike)
- **Minor Spill** — Chennai port, 0.8 km², ambiguous multi-candidate attribution

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Investigate

1. Select a test scenario from the sidebar
2. Click **Run Full Pipeline**
3. Explore the 5 tabs: Detection → Drift → Map → Attribution → Evidence

---

## 📁 Project Structure

```
MarineGuard/
├── app.py                              # Streamlit Investigation UI
├── config.py                           # Global configuration & constants
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── engines/                            # Core pipeline engines
│   ├── detection_engine.py             # SAR + U-Net → slick detection
│   ├── drift_engine.py                 # Backward drift → origin zone
│   ├── ais_engine.py                   # AIS filtering + trajectory
│   └── attribution_engine.py           # Multi-factor scoring + ranking
│
├── models/                             # AI model definitions
│   └── unet.py                         # U-Net architecture
│
├── utils/                              # Utility modules
│   ├── geo_utils.py                    # Geospatial functions
│   ├── visualization.py                # Map & chart generation
│   └── data_loader.py                  # Data loading + dataclasses
│
└── data/                               # Test data
    ├── generate_test_data.py           # Scenario generator
    └── test_scenarios/                 # Generated JSON scenarios
        ├── scenario_major_spill.json
        ├── scenario_no_spill.json
        └── scenario_minor_spill.json
```

---

## ⚖️ Attribution Scoring

Our Explainable Attribution Score replaces black-box probabilities with transparent, factor-based metrics:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Proximity** | 30% | Distance from vessel to estimated origin zone |
| **Temporal** | 25% | Alignment of vessel presence with release time window |
| **Trajectory** | 20% | Course/heading alignment with reverse drift path |
| **AIS Gap** | 15% | Suspicious transponder shutdown periods |
| **Behavioral** | 10% | Speed changes, loitering, course deviations |

---

## 🛡️ Innovation

- **Explainable Attribution Score** — No single-factor accusations
- **Uncertainty-Aware Origin Zone** — Realistic probable area, not a single point
- **Forensic Timeline** — Links slick detection → drift → vessel activity
- **Layered Evidence Map** — All data layers in one unified interface
- **"Why Ranked?" Panel** — Human-readable evidence for each factor

---

## 🧰 Technology Stack

| Category | Tools |
|----------|-------|
| **Satellite Data** | Sentinel-1 SAR |
| **Environmental** | ERA5 (wind), CMEMS (currents), Copernicus |
| **AI/ML** | U-Net (PyTorch), Segmentation |
| **Drift Modeling** | Lagrangian particle tracking (OpenDrift-inspired) |
| **Geospatial** | GeoPandas, Shapely, Rasterio |
| **Visualization** | Folium, Plotly |
| **Frontend** | Streamlit |
| **Backend** | Python, NumPy, Pandas |

---

## 📜 License

Built for Smart India Hackathon 2026 — Team Ocean X
