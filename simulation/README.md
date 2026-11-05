# OpenDSS Microgrid Simulation Package (Phase 2A - Milestone 1)

This directory contains the isolated OpenDSS microgrid simulation environment for the project **"AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids"**.

All simulation assets are strictly self-contained within this directory and do not modify or overwrite Phase 1 benchmark artifacts or trained model checkpoints.

---

## 1. Directory Structure

```text
simulation/
├── README.md                           # Package overview and usage guide
├── circuits/
│   └── master_microgrid.dss            # OpenDSS circuit definition script
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── circuit_builder.py              # MicrogridCircuitBuilder class (OpenDSS wrapper & physics)
│   └── runner.py                       # Milestone1Runner (snapshot and time-series validation)
└── tests/
    ├── __init__.py
    ├── test_minimal_circuit.py         # 6 automated validation tests (unittest)
    └── milestone1_results.json         # Serialized test results and physical logs
```

---

## 2. Microgrid Electrical Specification

The circuit models a low-voltage 3-phase microgrid equivalent to the **Data4Cyber** experimental testbed:

1. **Substation Transformer**:
   - Primary: 20.0 kV Medium Voltage (Delta connection)
   - Secondary: 0.4 kV Low Voltage (Wye-Grounded connection, 230 V Line-Neutral, 400 V Line-Line)
   - Rating: 100 kVA, $X_{HL} = 2.5\%$
2. **Distribution Feeders**:
   - 4 underground multiphase feeders connecting the secondary busbar (`SubstationBus`) to DER and load buses.
   - Cable parameters: $R_1 = 0.15\ \Omega/\text{km}$, $X_1 = 0.05\ \Omega/\text{km}$, length = 50 m each.
3. **Photovoltaic Inverter (`PVSystem.PV1`)**:
   - Fronius Symo equivalent: 36 kVA inverter rating, 20 kW baseline injection at nominal irradiance, unity power factor ($\cos\phi = 1.0$).
4. **Battery Energy Storage System (`Storage.BSS1`)**:
   - SMA Sunny Island equivalent: 25 kVA inverter rating, 20 kW rated power, 50 kWh capacity, initial 75% State of Charge (%stored).
5. **Distribution Loads**:
   - Industrial Load A: 7.0 kW active power, 1.5 kVAR inductive reactive power ($\cos\phi \approx 0.978$).
   - Industrial Load B: 8.0 kW active power, 2.0 kVAR inductive reactive power ($\cos\phi \approx 0.970$).

---

## 3. Physical Standards & Unit Conventions

- **Power Units**: Active power in **Watts (W)**, reactive power in **var**, apparent power in **VA** (converted from OpenDSS kW/kvar/kVA by multiplying by 1000.0).
- **Voltage Units**: RMS Line-to-Neutral in **Volts (V)** ($230\text{ V}$ nominal), Line-to-Line in **Volts (V)** ($400\text{ V}$ nominal).
- **Current Units**: Phase currents and neutral current in **Amperes (A)**.
- **Sign Conventions (Data4Cyber Benchmark Standard)**:
  - Generation / Export = **Negative ($-$)**
  - Load / Consumption = **Positive ($+$)**
  - Substation Secondary: Exporting to medium-voltage grid is **negative**, importing from grid is **positive**.
- **Neutral Current Definition**: Vector sum neutral current $I_N = |\vec{I}_1 + \vec{I}_2 + \vec{I}_3|$ matching Janitza UMG508 register 19018.

---

## 4. How to Run

Activate the project virtual environment and run the runner or test suite:

```bash
# Run Milestone 1 physical validation runner
.venv\Scripts\python.exe simulation/src/runner.py

# Run automated unittest suite (6 tests)
.venv\Scripts\python.exe -m unittest simulation/tests/test_minimal_circuit.py
```
