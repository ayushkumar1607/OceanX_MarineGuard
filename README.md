# 🛢️ MarineGuard — From Slick Detection to Maritime Evidence

> **Smart India Hackathon 2026 · Problem Statement 26143 · NTRO**
> *Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

**Team Ocean X** · SIH 2026

---

## 📖 Table of Contents

1. [Overview](#-overview)
2. [The MarineGuard Principle](#-the-marineguard-principle)
3. [System Architecture](#-system-architecture)
4. [End-to-End Pipeline](#-end-to-end-pipeline)
5. [The Three ML Models](#-the-three-ml-models)
6. [Detection Engine — U-Net](#-detection-engine--u-net)
7. [Drift Engine — Lagrangian Tracking](#-drift-engine--lagrangian-tracking)
8. [AIS Anomaly Model](#-ais-anomaly-model)
9. [Attribution Model](#-attribution-model)
10. [Tech Stack](#-tech-stack)
11. [Project Structure](#-project-structure)
12. [Installation](#-installation)
13. [Quick Start](#-quick-start)
14. [Model Performance](#-model-performance)
15. [Datasets](#-datasets)
16. [Screenshots](#-screenshots)
17. [Team](#-team)
18. [License](#-license)

---

## 🎯 Overview

**MarineGuard** is an end-to-end intelligent platform that investigates marine oil spills by fusing **three independent data sources**:

1. **Sentinel-1 SAR imagery** — detects the physical oil slick on the ocean surface
2. **Meteorological & oceanographic data** (ERA5 wind + CMEMS currents) — models how the slick drifted
3. **Historical AIS vessel traffic** — identifies which vessel was in the right place at the right time

The system detects oil slicks, traces them backward to their probable origin, filters thousands of vessels down to a handful of candidates, and produces an **explainable attribution score** for each — replacing black-box accusations with transparent, factor-based evidence.

### Why it matters

- **Marine oil spills** devastate ecosystems and coastal economies
- **Over 90% of illegal discharges** go unattributed because satellites rarely capture the polluting act itself
- **MarineGuard closes that gap** by combining physics-based drift modelling with machine learning attribution

---

## 🧭 The MarineGuard Principle

> ***"Find the vessel that best fits the incident — not just the closest."***

We never accuse a vessel based on proximity alone. Every attribution requires **multiple corroborating factors**: timing, trajectory, AIS behaviour, and physical consistency with the drift model.

---

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph INPUT["📥 INPUT LAYER"]
        SAR["🛰️ Sentinel-1 SAR<br/>Image (PNG/TIFF)"]
        ENV["🌊 Environmental Data<br/>ERA5 Wind + CMEMS Currents"]
        AIS["🚢 Historical AIS<br/>Vessel Traffic JSON"]
    end

    subgraph ML["🧠 ML LAYER"]
        UNET["U-Net<br/>1.9M params<br/>Oil Slick Segmentation"]
        ANOM["Random Forest<br/>AIS Anomaly Detection"]
        ATTR["Gradient Boosting<br/>Attribution Scoring"]
    end

    subgraph ENGINES["⚙️ PROCESSING LAYER"]
        DET["Detection Engine<br/>SAR → Slick mask"]
        DRIFT["Drift Engine<br/>Backward Lagrangian<br/>Particle Tracking"]
        AISENG["AIS Engine<br/>Spatial-temporal<br/>Filtering + Gap Detection"]
        ATTRENG["Attribution Engine<br/>Multi-factor<br/>Evidence Scoring"]
    end

    subgraph OUTPUT["📤 OUTPUT LAYER"]
        MAP["🗺️ Interactive Maps"]
        EVID["📋 Evidence Panel<br/>Why Ranked?"]
        RANK["📊 Ranked Vessels<br/>Attribution Scores"]
    end

    SAR --> UNET
    UNET --> DET
    ENV --> DRIFT
    DET --> DRIFT
    DRIFT --> AISENG
    AIS --> AISENG
    AISENG --> ANOM
    ANOM --> ATTRENG
    DRIFT --> ATTRENG
    ATTRENG --> ATTR
    ATTRENG --> RANK
    ATTRENG --> EVID
    DET --> MAP
    DRIFT --> MAP
    AISENG --> MAP

    style INPUT fill:#1e3a5f,stroke:#4ECDC4,color:#fff
    style ML fill:#2d1b4e,stroke:#a855f7,color:#fff
    style ENGINES fill:#1e293b,stroke:#45B7D1,color:#fff
    style OUTPUT fill:#14532d,stroke:#34C759,color:#fff
    style UNET fill:#EE4C2C,color:#fff
    style ANOM fill:#F7931E,color:#fff
    style ATTR fill:#F7931E,color:#fff


    flowchart LR
    subgraph S1["1️⃣ DETECT"]
        A1[Sentinel-1 SAR Image] --> A2[U-Net Segmentation]
        A2 --> A3[Slick Mask +<br/>Area + Centroid<br/>+ Confidence]
    end

    subgraph S2["2️⃣ TRACE"]
        A3 --> B1[Backward Lagrangian<br/>Particle Tracking]
        ENV2[ERA5 Wind<br/>CMEMS Currents] --> B1
        B1 --> B2[Origin Zone<br/>+ Release Time Window<br/>+ Uncertainty Ellipse]
    end

    subgraph S3["3️⃣ CORRELATE"]
        B2 --> C1[Spatial Filter<br/>Radius: 50 km]
        AIS2[AIS Traffic] --> C1
        C1 --> C2[Temporal Filter<br/>± 6 h]
        C2 --> C3[Trajectory Analysis<br/>+ AIS Gap Detection]
        C3 --> C4[Random Forest<br/>Anomaly Scoring]
    end

    subgraph S4["4️⃣ RANK"]
        C4 --> D1[Factor Scoring]
        B2 --> D1
        D1 --> D2[Proximity 30%<br/>Temporal 25%<br/>Trajectory 20%<br/>AIS Gap 15%<br/>Behavioral 10%]
        D2 --> D3[Gradient Boosting<br/>Attribution Model]
    end

    subgraph S5["5️⃣ EXPLAIN"]
        D3 --> E1[Ranked Vessels]
        D3 --> E2[Evidence Panel]
        D3 --> E3[Interactive Map]
    end

    style S1 fill:#0f172a,stroke:#FF6B35,color:#fff
    style S2 fill:#0f172a,stroke:#45B7D1,color:#fff
    style S3 fill:#0f172a,stroke:#FF9500,color:#fff
    style S4 fill:#0f172a,stroke:#a855f7,color:#fff
    style S5 fill:#0f172a,stroke:#34C759,color:#fff

    sequenceDiagram
    autonumber
    participant U as 👤 User
    participant A as 🖥️ Streamlit UI
    participant D as 📡 Detection Engine
    participant W as 🌊 Drift Engine
    participant S as 🚢 AIS Engine
    participant R as 📊 Attribution Engine

    U->>A: Upload SAR + Env + AIS
    A->>A: Validate 3-input set
    A->>D: Run detection
    D->>D: U-Net inference
    D-->>A: Slick mask + confidence + area

    alt Spill Detected
        A->>W: Trace origin
        W->>W: Backward Lagrangian tracking
        W->>W: Compute uncertainty zone
        W-->>A: Origin + release window

        A->>S: Correlate vessels
        S->>S: Spatial filter (50 km radius)
        S->>S: Temporal filter (± 6 h)
        S->>S: Trajectory + AIS gap analysis
        S->>S: ML anomaly scoring
        S-->>A: Ranked vessel candidates

        A->>R: Attribute
        R->>R: Score 5 factors per vessel
        R->>R: Gradient Boosting model
        R-->>A: Attribution scores + evidence

        A-->>U: Show maps + ranked vessels + evidence
    else No Spill
        A-->>U: Show false-positive explanation
    end
