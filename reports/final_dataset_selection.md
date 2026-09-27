# Final Smart Grid Dataset Selection Report

## 1. Comprehensive Multi-Dataset Comparative Evaluation

| Evaluation Criteria | DS-01: binaryAllNatural | DS-02: IEEE SGSMA PMU | DS-04: DNP3 SCADA | DS-05: IEC 104 OT | DS-06: Data4Cyber DER |
|---|---|---|---|---|---|
| **Domain Telemetry** | Physical + Cyber logs | Physical PMU + Comms | Pure Cyber Network | Pure Cyber SCADA | **True Cyber-Physical Multi-Modal** |
| **Physical Grid Telemetry** | 116 Phasor Relay features | 14 PMU waveform channels | None | None | **PV, Battery SoC, AC Meters, Volts, Freq** |
| **Cyber Protocol Telemetry** | Snort alert counts | Comms drop indicator | DNP3 Flow metrics | IEC 104 ASDU/COT | **Modbus Regs, MQTT Price, Network Taps** |
| **Temporal Granularity** | Static Event Snapshots | 30 Hz Waveform | Tabular flow statistics | Windowed aggregations | **Strict 1.0 s Aligned Grid** |
| **Attack Diversity** | 1 aggregated attack class | 7 physical & cyber events | 10 DNP3 attack types | 11 IEC 104 attack types | **4 Attack Families, 15 Attack Phases** |
| **Unseen Attack Generalization** | Unsupported | Cyber drop hold-out | Single-protocol hold-out | Single-protocol hold-out | **Cross-Family Zero-Day Generalization** |
| **Threat-Hunting Suitability** | Low | Moderate | Moderate | Moderate | **Superior (Packet capture, phase timestamps)** |
| **Autonomous Defense Ready** | Low | Low (Open-loop) | Low | Low | **High (PV inverter control, battery curtailment)** |

---

## 2. Dataset Specific Strengths & Limitations

### DS-01: `binaryAllNaturalPlusNormalVsAttacks`
- **Strengths**: Large sample size (78,377 rows); multi-relay physical transmission system measurements; good for static tabular baselines.
- **Limitations**: Attack subtypes are collapsed into a single `Attack` label, making multiclass classification impossible; lack of true time-series timestamps prevents dynamic sequential modeling.

### DS-02: `PMU_Data_with_Anomalies_and_Events` (IEEE SGSMA 2024)
- **Strengths**: High-frequency 30 Hz synchrophasor waveform dynamics; ideal for 1D-CNN, LSTM, and CNN-LSTM models; represents real-time PMU measurements on IEEE 39-bus New England system.
- **Limitations**: Severe class imbalance for physical faults (149 fault samples vs 133k normal); cyber telemetry is limited to communication drop indicators rather than protocol-level payload semantics.

### DS-04: `CICFlowMeter_Balanced_DNP3_SCADA` & DS-05: `Balanced_IEC104_Train_Test_CSV_Files`
- **Strengths**: Balanced multi-class attack taxonomies; rich industrial protocol features (DNP3 and IEC 60870-5-104 ASDU/COT semantics); high classification accuracy.
- **Limitations**: Completely lack physical grid measurements (zero electrical telemetry, zero voltage/frequency impact). Unsuitable for holistic cyber-physical security analysis.

### DS-06: `data4cyber_dataset`
- **Strengths**: State-of-the-art cyber-physical multi-modal testbed for Distributed Energy Resources (DERs); contains both physical grid metrics (solar generation, battery energy storage State of Charge, AC power/current/voltage) and cyber metrics (Modbus registers, MQTT price signals, PCAP network captures); pristine benign baseline (Scenario S0); rich multi-phase attack campaigns.
- **Limitations**: Moderate sample count (14,354 samples across S0-S6); requires careful leakage prevention (stripping internal `Attacker.*` execution logs).

---

## 3. SELECTED DATASET FOR FINAL PROJECT

### Primary Dataset: **`data4cyber_dataset`**
*(with **`PMU_Data_with_Anomalies_and_Events`** as Secondary High-Frequency Waveform Benchmark)*

### Concrete Reasons for Selection:
1. **Holistic Cyber-Physical Multi-Modality**: Modern smart grid threats (e.g., Industroyer2, Stuxnet) manipulate cyber commands to induce physical grid instability. `data4cyber_dataset` is the only dataset that captures both the cyber exploit (Modbus/MQTT tampering) and its direct physical grid impact (battery overcharge, PV curtailment override, voltage fluctuations).
2. **Superior Zero-Day / Unseen Attack Research**: Contains 4 distinct attack families (`industroyer_modbus`, `mitm_false_data_injection`, `mqtt_price_signal_manipulation`). This allows training on one family (e.g. Modbus manipulation) and rigorously testing zero-shot generalization against completely unseen attack mechanisms (e.g. MitM FDI or supply-chain MQTT attacks).
3. **Pristine Benign Baseline for Anomaly Detection**: Scenario S0 provides an uncompromised benign baseline, allowing Autoencoders and One-Class classifiers to learn normal physical-cyber invariants without label contamination.
4. **Foundation for Downstream Major Project Phases**: The rich multi-modal logs (including PCAPs, IP mappings, and phase semantics) directly support our upcoming project phases: Threat Hunting, Risk Assessment, and Safe Reinforcement Learning for Autonomous Grid Defense.
