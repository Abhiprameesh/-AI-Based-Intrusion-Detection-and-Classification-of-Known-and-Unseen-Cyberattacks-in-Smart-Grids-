# Smart Grid Intrusion Detection System (IDS): Complete Project & Manual Troubleshooting Guide

**Project Title:** AI-Based Intrusion Detection and Classification of Known and Unseen Cyberattacks in Smart Grids  
**Author:** Final Year Major Project  
**Target Audience:** Students, Evaluators, Examiners, and Developers  

---

## Table of Contents
1. [Project Overview in Simple Words (Layman's Guide)](#1-project-overview-in-simple-words-laymans-guide)
2. [The Three-Stage Defense Architecture](#2-the-three-stage-defense-architecture)
3. [The Cyber-Physical Simulation & 9 Scenarios (S0–S8)](#3-the-cyber-physical-simulation--9-scenarios-s0s8)
4. [Step-by-Step Execution Guide (How to Run Everything)](#4-step-by-step-execution-guide-how-to-run-everything)
5. [Known Issues Encountered & Why They Happened](#5-known-issues-encountered--why-they-happened)
6. [How to Fix These Issues Manually (Without AI Agents)](#6-how-to-fix-these-issues-manually-without-ai-agents)
7. [Examiner & Viva Q&A Guide](#7-examiner--viva-qa-guide)

---

## 1. Project Overview in Simple Words (Layman's Guide)

### What is a Smart Grid?
An electrical grid traditionally delivers electricity from a big power plant to your home. A **Smart Grid** adds computer chips, sensors (smart meters), and communication networks (like MQTT or Modbus) to coordinate modern renewable energy sources:
- **Rooftop Solar (PV Inverters)**
- **Battery Storage Systems (BSS)**
- **Dynamic consumer loads**

### The Threat: Why Machine Learning is Needed
Because smart grids connect to the internet and local networks, hackers can attack them. Hackers can:
1. **Tamper with meter readings (False Data Injection / MitM):** Lie about how much solar power or battery charge is available so the grid operator makes bad decisions.
2. **Send illegal commands to physical switches (Actuator Control / Industroyer):** Remotely disconnect solar panels or drain large batteries, causing blackouts or voltage instability.
3. **Freeze sensors (Denial of Service / Sensor Dropout):** Replay old readings so operators don't see sudden equipment failures.

### What This Project Built
Most student machine learning projects take a simple CSV file, split it into 80/20 train/test, and print "99% accuracy". That does **not** work in the real world because electrical grids follow physical laws (Kirchhoff's laws, Ohm's law, power flow equations).

This project built a **complete, closed-loop cyber-physical framework**:
1. We used genuine machine learning models trained on real laboratory testbed data (**Data4Cyber dataset**).
2. We built a **physics simulator** using the industry-standard power system engine (**OpenDSS**).
3. We simulated 9 realistic operational scenarios (normal operations, hacker attacks, and sudden weather changes).
4. We evaluated the models against the physical grid, uncovered real-world problems like **Distribution/Domain Shift**, and built an interactive dashboard to monitor the entire grid in real time.

---

## 2. The Three-Stage Defense Architecture

Think of the 3-stage defense system like airport security:

```
[Incoming 150-Feature Meter Telemetry]
                  │
                  ▼
   ┌─────────────────────────────┐
   │  STAGE 1: BINARY XGBOOST    │  --> "Is this sample benign or an attack?"
   └──────────────┬──────────────┘
                  │ (If Attack Detected or Diagnostic Flag Raised)
                  ▼
   ┌─────────────────────────────┐
   │  STAGE 2: MULTICLASS RF     │  --> "What type of known attack is this?"
   └──────────────┬──────────────┘      (Industroyer? MitM FDI? MQTT tampering?)
                  │
                  ▼
   ┌─────────────────────────────┐
   │  STAGE 3: DEEP AUTOENCODER  │  --> "Is this a brand new (zero-day) attack?"
   └─────────────────────────────┘      (Reconstruction MSE residual analysis)
```

### Stage 1: The Fast Filter (XGBoost Classifier)
- **Role:** Binary Attack Detection.
- **Input:** 150 raw electrical measurements (voltages, active powers, reactive powers, currents, power factors across all 4 sub-meters).
- **Decision:** Outputs an attack probability between `0.0` and `1.0`.
  - **Primary Threshold ($\tau = 0.50$):** High-confidence attack alert.
  - **Diagnostic Threshold ($\tau = 0.05$):** Sensitive alert used to catch subtle, low-magnitude anomalies.

### Stage 2: The Attack Profiler (Random Forest Classifier)
- **Role:** Multiclass Threat Attribution.
- **Trained Classes:**
  - `Class 0`: Benign Normal
  - `Class 1`: Industroyer (actuator trip / switch commands)
  - `Class 2`: MitM FDI (False Data Injection scaling/biases)
  - `Class 3`: MQTT Tampering (command manipulation)
- **Why it matters:** Grid operators need to know whether the physical hardware is being turned off or if only the network packets are being modified.

### Stage 3: The Zero-Day Safety Net (Deep Autoencoder)
- **Role:** Unseen Attack & Anomaly Detection.
- **Architecture:** PyTorch Neural Network with an encoder that compresses 150 features into a 16-dimensional bottleneck, and a decoder that reconstructs them back to 150 features.
- **How it works:** The Autoencoder was trained **only on normal (benign) grid data**. When normal data passes through, the reconstruction error (Mean Squared Error - MSE) is very low ($< 0.364$). When a novel, unseen cyberattack arrives, the network fails to reconstruct it, the MSE spikes above the threshold, and a **Zero-Day Attack Alarm** is triggered.

---

## 3. The Cyber-Physical Simulation & 9 Scenarios (S0–S8)

All scenarios run for **150 seconds (150 discrete 1-second timesteps)** using OpenDSS.

| Scenario Code | Scenario Name | Category | What It Does In Plain Words |
| :---: | :--- | :---: | :--- |
| **S0** | **Benign Baseline** | Normal | Grid operates normally. PV generates clean solar power, battery idles, loads consume power. Zero attacks. |
| **S1** | **FDI Scaling (BSS)** | Known Attack | Hacker sits on the battery communication wire and scales reported battery power down by 50% ($0.5\times$). Physical battery is fine; reported telemetry is false. |
| **S2** | **FDI Bias (Substation)** | Known Attack | Hacker adds a $+5.0\text{ kW}$ false bias to substation import telemetry packets. |
| **S3** | **Industroyer PV Trip** | Known Attack | Hacker sends an unauthorized command directly to the solar inverter switch, instantly shutting off the PV system ($0\text{ kW}$). Physical voltage drops. |
| **S4** | **Industroyer BSS Discharge** | Known Attack | Hacker forces the battery to discharge at maximum $15\text{ kW}$ while hacking the reported State of Charge (SOC) meter to report a fake normal 20%. |
| **S5** | **Load-B Telemetry Freeze** | Known Attack | Hacker freezes the Load-B meter for 70 seconds ($t = 40$ to $110$), constantly replaying the stale reading from step 40. |
| **S6** | **Unseen Stealthy Ramp FDI** | Zero-Day | Subtle attack: slowly ramps solar telemetry up at $0.05\text{ kW/s}$ to evade Stage 1 binary detection. |
| **S7** | **Unseen Coordinated Masking**| Zero-Day | Smart coordinated attack: hacks both substation and battery simultaneously so that Kirchhoff's power sum appears balanced. |
| **S8** | **Benign Shading Disturbance** | Benign Control | **Crucial Scientific Control:** A fast-moving heavy cloud drops solar generation by 60% in 5 seconds. Proves that the IDS does **not** sound a false alarm during natural weather changes! |

---

## 4. Step-by-Step Execution Guide (How to Run Everything)

All commands below are designed for **Windows PowerShell** using the project's dedicated virtual environment (`.venv`).

### Step 4.1: Open PowerShell and Navigate to Project
```powershell
cd "c:\ALLMLPROJ\MAINMLPROJS\MAJOR PROJECT"
```

### Step 4.2: Verify the Virtual Environment
Verify Python and required packages are present:
```powershell
.venv\Scripts\python.exe --version
# Expected: Python 3.11.x or 3.12.x

.venv\Scripts\streamlit.exe --version
# Expected: Streamlit 1.66.1 (or newer)
```

### Step 4.3: Run the Automated Unit Test Suite (33 Tests)
Before launching the UI, always verify that the simulation physics, power balance, and frozen models pass all tests:
```powershell
.venv\Scripts\python.exe -m unittest discover simulation/tests -v
```
**Expected Output:**
```
Ran 33 tests in ~11s
OK
```

### Step 4.4: Launch the Interactive Streamlit Dashboard
To run the full graphical user interface:
```powershell
.venv\Scripts\streamlit.exe run dashboard/app.py
```
- Streamlit will open your default browser automatically at `http://localhost:8501`.
- If it does not open automatically, open your browser and navigate to `http://localhost:8501`.

### Step 4.5: Run the Verification Script (Optional Sanity Check)
To verify that all scenario simulations and cryptographic hashes are 100% intact:
```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'simulation/src'); import verify_evidence_read_only; verify_evidence_read_only.verify()"
```

---

## 5. Known Issues Encountered & Why They Happened

During this project, we encountered several real-world technical and scientific challenges. Understanding these will help you answer examiner questions with confidence.

### Issue 1: "Domain Shift" (Distribution Shift)
- **Symptom:** When Stage 1 XGBoost (trained on laboratory Data4Cyber data) was initially tested on the OpenDSS simulated grid, it detected 0% of attacks on S3 under baseline conditions.
- **Root Cause:** In the physical laboratory testbed, the voltage was stiff at $234.8\text{ V}$ with zero reactive power ($Q = 0$). In the standard OpenDSS microgrid simulation, the nominal voltage was $230.6\text{ V}$ and loads drew inductive reactive power ($Q > 0$). Because the ML model was trained on $234.8\text{ V}$, it viewed $230.6\text{ V}$ as abnormal, shifting predicted probabilities downward.
- **Solution (Conditions C0–C3):** We conducted a controlled 4-way experiment:
  - `C0`: Default simulation baseline ($230.6\text{ V}, Q \neq 0$)
  - `C1`: Voltage alignment only ($234.8\text{ V}$)
  - `C2`: Reactive power alignment only ($Q = 0$)
  - `C3`: Combined alignment ($234.8\text{ V}, Q = 0$)
  Under C3, attack detection probability jumped by **$+61.5\%$**, proving the model was working correctly and highlighting the cyber-physical domain shift phenomenon.

### Issue 2: Scenario S5 (Load-B Freeze) Showing 0 W Discrepancy
- **Symptom:** In S5, the hacker freezes Load-B telemetry from step 40 to 110, but the difference between reported telemetry and true physical load was $0.000\text{ W}$.
- **Root Cause:** In the underlying Data4Cyber dataset, the laboratory load profile `Profile.load2_kw` was recorded during steady-state operation, meaning it was constant at $13.654\text{ kW}$ across all timesteps. Freezing a constant value at step 40 means the frozen reading ($13.654\text{ kW}$) is identical to the physical load ($13.654\text{ kW}$) at every step.
- **Solution:** We updated the dashboard to visually show the frozen sensor curve and shade the attack window ($t = 40$ to $110$), explaining to the user that **zero numerical discrepancy was caused by physical load invariance, not because the attack failed**.

### Issue 3: Plotly Confusion Matrix TypeError Crash
- **Symptom:** Opening the dashboard on Tab 3 crashed with:
  `TypeError: update_layout() got multiple values for keyword argument 'margin'`
- **Root Cause:** In `dashboard/components.py`, the layout was updated with `**DARK_LAYOUT` (which already contained `margin=dict(l=40, r=40, t=50, b=40)`), while also passing `margin=dict(...)` as an explicit argument. Plotly rejected the duplicate keyword argument.
- **Solution:** In `dashboard/components.py`, merged the dictionary before passing:
  ```python
  layout = dict(DARK_LAYOUT)
  layout["margin"] = dict(l=60, r=40, t=60, b=60)
  fig.update_layout(layout)
  ```

### Issue 4: Sidebar Scenario Desynchronization
- **Symptom:** Selecting a scenario in the sidebar dropdown did not immediately update the graphs until "Run Simulation" was clicked, confusing users.
- **Root Cause:** Streamlit reruns the script on dropdown interaction, but the underlying simulation results remained bound to the old `st.session_state["simulation_results"]`.
- **Solution:** Added Quick Demonstration buttons (S0, S3, S5, S8) that atomically simulate and update session state immediately, plus added a prominent badge indicating:
  `CURRENTLY DISPLAYED RESULTS: [S3] Industroyer PV Trip` with an alert if the sidebar dropdown doesn't match the current display.

### Issue 5: Discrepancy Plot Omission (Missing Substation & Load-B)
- **Symptom:** Scenarios S2 (Substation Bias) and S7 (Substation Masking) displayed "No cyber discrepancy detected" even though the hacker was injecting $+5\text{ kW}$ or $+10\text{ kW}$ false data.
- **Root Cause:** `plot_cyber_physical_discrepancy()` in `dashboard/components.py` only checked `pv_disc` and `bss_disc`, completely ignoring substation and load meters.
- **Solution:** Expanded `evaluate_cyber_physical_discrepancy()` to inspect all four measurement channels: PV, BSS, Load-B, and Substation.

---

## 6. How to Fix These Issues Manually (Without AI Agents)

If you ever need to inspect, modify, or troubleshoot the codebase yourself on another machine or after cloning, follow these instructions.

### 6.1 How to Check File Hashes Manually
If you want to prove to an evaluator that the Phase 1 machine learning models were **never modified or retrained**, run this simple Python one-liner in PowerShell:
```powershell
.venv\Scripts\python.exe -c "
import hashlib
for path in [
    'phase_1/results/data4cyber_dataset/binary/models/XGBoost.joblib',
    'phase_1/results/data4cyber_dataset/binary/models/Random_Forest.joblib',
    'phase_1/results/data4cyber_dataset/multiclass/models/Random_Forest_multiclass.joblib',
    'phase_1/results/data4cyber_dataset/anomaly/recovery_v1/Autoencoder.pt'
]:
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    print(f'{path} -> {h[:16]}...')
"
```
**Expected Hashes (first 16 chars):**
- `XGBoost.joblib`: `f14e1c87d3640bcc...`
- `Random_Forest.joblib`: `6a66af5e955bc0a5...`
- `Random_Forest_multiclass.joblib`: `8987a92549af9433...`
- `Autoencoder.pt`: `8df8b7333d93ff58...`

### 6.2 How to Run a Single Scenario Directly in Python
If Streamlit fails to launch or you want to inspect a scenario's numbers directly without the UI:
```python
# Save as run_test.py and execute with: .venv\Scripts\python.exe run_test.py
import sys
sys.path.insert(0, "simulation")
from src.ids_core import IDSInferenceEngine

# Initialize the unified simulation & inference engine
engine = IDSInferenceEngine()

# Run Scenario S3 (150 steps)
features_df, metadata_df = engine.run_scenario_simulation("S3_industroyer_pv_trip", num_steps=150)
results = engine.run_full_inference(features_df, metadata_df)

print("Stage 1 Max Probability:", results["stage1"]["probabilities"].max())
print("Stage 2 Predicted Classes (mode):", results["stage2"]["predicted_classes"].mode()[0])
print("Stage 3 Mean Autoencoder MSE:", results["stage3"]["reconstruction_mse"].mean())
```

### 6.3 How to Change the Dashboard Thresholds Manually
If an examiner asks: *"What happens if you change the diagnostic threshold from 0.05 to 0.10?"*
1. Open [`dashboard/components.py`](file:///C:/ALLMLPROJ/MAINMLPROJS/MAJOR PROJECT/dashboard/components.py).
2. Locate `plot_stage1_probabilities()` around line 660.
3. Find the threshold lines:
   ```python
   # Primary operating threshold
   fig.add_hline(y=0.50, line_dash="dash", line_color="#ef4444", annotation_text="Primary Threshold (tau = 0.50)")
   # Diagnostic threshold
   fig.add_hline(y=0.05, line_dash="dot", line_color="#f59e0b", annotation_text="Diagnostic Threshold (tau = 0.05)")
   ```
4. Change `0.05` to any value you want (e.g. `0.10`), save the file, and refresh your browser.

### 6.4 How to Fix Port Conflicts (If Port 8501 is Busy)
If Streamlit says `Port 8501 is already in use`, run it on another port:
```powershell
.venv\Scripts\streamlit.exe run dashboard/app.py --server.port 8505
```

---

## 7. Examiner & Viva Q&A Guide

Here are the most common questions examiners will ask about this project, along with the precise answers you should provide:

### Q1: "Why did you use a 3-Stage IDS instead of a single Deep Learning model?"
> **Answer:** *"In smart grids, speed, explainability, and safety are paramount. A single deep network behaves like a black box. Our three-stage approach provides defense-in-depth:  
> 1. Stage 1 (XGBoost) provides microsecond-level binary screening.  
> 2. Stage 2 (Random Forest) attributes attacks to known operational threats (like Industroyer or MitM) so operators take the right action.  
> 3. Stage 3 (Autoencoder) serves as a specialized zero-day detector. Separating detection, attribution, and anomaly discovery makes the system resilient and transparent."*

### Q2: "Why was Stage 1 recall initially low on the OpenDSS simulation?"
> **Answer:** *"That was one of the central research discoveries of this project: **cyber-physical distribution shift**. The machine learning models were pre-trained on laboratory testbed data where the nominal bus voltage was stiff at $234.8\text{ V}$ with zero reactive power ($Q=0$). When tested against a dynamic simulation operating at $230.6\text{ V}$ with inductive loads ($Q>0$), the feature distributions shifted. By proving this in Milestone 4.5 through Conditions C0 to C3 (where adjusting operating conditions restored detection probabilities by $+61.5\%$), we demonstrated that power system physics must be considered when deploying ML in power grids."*

### Q3: "In Scenario S5, why is the numerical discrepancy between reported and physical load 0 W?"
> **Answer:** *"In Scenario S5, the attacker executes a sensor freeze attack, locking the telemetry to the value captured at timestep 40. In our underlying laboratory driving dataset, Load B was operating in steady-state ($13.654\text{ kW}$ constant). Because the physical load did not fluctuate, the frozen reported value matched the physical value. This was a vital finding: it proves that relying purely on numerical discrepancy can miss sensor freeze attacks on constant loads, which is why Stage 3 Autoencoders and timing checks are necessary."*

### Q4: "How do you distinguish a real cyberattack from a cloud passing over a solar panel?"
> **Answer:** *"We specifically tested this using **Scenario S8 (Benign Cloud Cover Shading)**, where solar generation dropped by 60% in 5 seconds. Because cloud shading follows natural solar irradiance physics, the relationship between solar input, panel temperature, and output voltage remains physically consistent. The Stage 1 XGBoost model correctly recognized this as benign ($\max P = 0.0447 < 0.05$), producing **0 false alarms**."*

### Q5: "Are the machine learning models retrained on OpenDSS data?"
> **Answer:** *"No. To maintain strict scientific integrity, all models and preprocessors from Phase 1 remain 100% frozen. We verified their SHA-256 cryptographic hashes before and after all experiments to guarantee that no weights or decision boundaries were altered."*

---

## 8. Summary Checklist Before Final Submission

- [x] Virtual environment `.venv` verified with Python and Streamlit.
- [x] All 33 automated tests passing (`unittest discover simulation/tests`).
- [x] Streamlit dashboard verified on all 4 tabs.
- [x] Quick demonstration buttons verified for S0, S3, S5, and S8.
- [x] All 11 canonical artifact hashes verified bit-for-bit intact.
- [x] Git ignore configured to keep experiment reports and caches out of Git.
- [x] Examiner presentation and viva script prepared.

