# Smart Grid Intrusion Detection System (IDS) Dashboard

An interactive, research-grade cybersecurity demonstration dashboard for the project:  
**"AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids"**

---

## 1. Quick Launch Instructions

Launch the dashboard from the project root directory using the virtual environment:

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 2. Architecture Overview

```
MAJOR PROJECT/
├── dashboard/
│   ├── app.py            # Main Streamlit application and tab layouts
│   ├── components.py     # UI components, custom dark CSS theme, Plotly charts
│   ├── utils.py          # Cached inference engine, loaders, system health check
│   └── README.md         # Launch guide and demonstration protocol
├── simulation/
│   ├── src/
│   │   ├── ids_core.py   # Unified in-memory three-stage IDS inference engine
│   │   ├── scenario_runner.py  # OpenDSS QSTS power flow and attack injector
│   │   ├── telemetry_exporter.py # 150-feature schema alignment engine
│   │   └── circuit_builder.py    # OpenDSS microgrid physical model
│   └── results/          # Canonical, untouched milestone artifacts
└── phase_1/results/      # Frozen, bit-audited machine learning models
```

### Defense-in-Depth Pipeline
1. **Stage 1 (Binary Detection):** Frozen XGBoost evaluates whether the grid is under cyberattack ($\tau = 0.50$ primary, $\tau = 0.05$ diagnostic sensitivity).
2. **Stage 2 (Known Attack Classification):** Frozen Random Forest classifies intrusions into known threat profiles: Benign (0), Industroyer (1), MITM FDI (2).
3. **Stage 3 (Unseen Anomaly Detection):** Recovered Deep Autoencoder detects zero-day attacks via reconstruction residual error ($\tau = 0.36402026$).

---

## 3. Recommended Demonstration Walkthrough

For the final project evaluation with the guide/examiners, follow this sequence:

### Step 1: Baseline Verification (S0)
- In the sidebar, select **"🟢 S0: Benign"** and click **"Run Simulation"**.
- View **Tab 1 (Grid Monitoring)**: Observe nominal 230.6 V bus voltages, balanced active power, and stable battery SOC.
- View **Tab 3 (Three-Stage IDS)**: Confirm Stage 1 attack probability is low ($P \approx 0.012$), Stage 2 identifies Benign (Class 0), and Stage 3 shows nominal simulation MSE.

### Step 2: Actuator Attack Demonstration (S3 - Industroyer PV Trip)
- In the sidebar, click **"🔴 S3: PV Trip"** (the scenario will execute and load immediately).
- View **Tab 1 (Grid Monitoring)**: Observe complete curtailment of PV solar generation during steps 40–110.
- View **Tab 3 (Three-Stage IDS)**:
  - **Stage 1:** Observe that in live baseline simulation (C0), peak probability reaches $0.0447$ ($< 0.05$). Under Milestone 4.5 C3 combined alignment, probability rose to $0.0722$, detecting $71/71$ attack steps ($100\%$ scenario recall, $13.22\%$ suite recall).
  - **Stage 2:** Observe that $71/71 = 100.00\%$ of active PV trip attack steps are correctly classified as **Industroyer (Class 1)**. Note the distinction that multi-scenario Industroyer recall across both S3 and S4 is $70.42\%$ ($100/142$) due to S4's spoofed normal SOC telemetry.

### Step 3: Cyber Telemetry Manipulation (S5 - Load-B Telemetry Freeze)
- In the sidebar, select **"🟠 S5: Freeze"** and click **"Run Simulation"**.
- View **Tab 1 (Grid Monitoring)**: Observe that physical power flow continues normally, but reported telemetry freezes.
- View **Tab 3 (Three-Stage IDS)**: Observe that single-meter cyber manipulation evades Stage 1 ($P < 0.02$) and is correctly noted as out-of-scope for Stage 2.

### Step 4: Benign Physical Disturbance (S8 - Cloud Cover Shading)
- In the sidebar, select **"🟡 S8: Shading"** and click **"Run Simulation"**.
- View **Tab 1 (Grid Monitoring)**: Observe 70% irradiance dip in solar generation.
- View **Tab 3 (Three-Stage IDS)**: Demonstrates the critical scientific finding that physical solar drops trigger the same probability spike ($0.0447$–$0.0722$) as physical PV curtailment, revealing that XGBoost conflates solar power loss with cyber intrusions.

### Step 5: Research Analysis (Tab 4)
- Navigate to **Tab 4 (Research Analysis)**: Review the cross-domain comparison between the native Data4Cyber laboratory benchmark ($>99\%$ accuracy) and OpenDSS simulation conditions (C0 to C3).
- Discuss the five key findings regarding domain shift, voltage threshold boundaries, and reconstruction residual compression.

---

## 4. Scientific Guardrails Enforced
- **Zero Retraining:** All Phase 1 models and scalers remain 100% frozen.
- **Non-Destructive:** The dashboard executes simulations and predictions strictly in memory; canonical baseline files are never overwritten.
- **Research Honesty:** Model limitations, low recall on cyber attacks, and cross-domain distribution shifts are reported transparently.

