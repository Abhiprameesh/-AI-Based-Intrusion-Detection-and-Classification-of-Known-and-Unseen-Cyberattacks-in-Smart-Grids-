"""
Circuit Builder and Physical Measurement Extractor for OpenDSS Microgrid.
Provides low-level OpenDSS execution, circuit compilation, and physical measurement extraction.
All power values are exported in Watts (W), var, and VA to match Data4Cyber units.
"""

import os
import cmath
from typing import Dict, Any, Tuple, Optional
import opendssdirect as dss


class MicrogridCircuitBuilder:
    """
    Manages the OpenDSS 5-bus microgrid circuit and extracts physical electrical measurements.
    """

    METER_LOCATIONS = {
        "Substation-AC-Meter": {"element": "Transformer.Substation", "bus": "SubstationBus", "terminal": 2},
        "PV-Inverter-AC-Meter": {"element": "PVSystem.PV1", "bus": "PVBus", "terminal": 1},
        "BSS-Inverter-AC-Meter": {"element": "Storage.BSS1", "bus": "BSSBus", "terminal": 1},
        "Load-A-AC-Meter": {"element": "Load.LoadA", "bus": "LoadABus", "terminal": 1},
        "Load-B-AC-Meter": {"element": "Load.LoadB", "bus": "LoadBBus", "terminal": 1},
    }

    def __init__(self, circuit_file: str = None):
        if circuit_file is None:
            # Default to circuit file in simulation/circuits/
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            circuit_file = os.path.join(pkg_dir, "circuits", "master_microgrid.dss")
        self.circuit_file = circuit_file
        self._compiled = False

    def build_circuit(
        self,
        initial_soc_percent: Optional[float] = None,
        reserve_pct: Optional[float] = None,
        slack_pu: Optional[float] = None,
        load_a_kvar: Optional[float] = None,
        load_b_kvar: Optional[float] = None,
    ) -> bool:
        """
        Compiles the master OpenDSS circuit file.
        initial_soc_percent: Optional initial battery charge in percent (e.g. 20.0 for Data4Cyber match, or 75.0 default).
        reserve_pct: Optional minimum reserve cutoff percentage (defaults to 0.0 if initial_soc_percent <= 20.0).
        slack_pu: Optional grid slack bus voltage in per-unit (default from DSS file is 1.000).
        load_a_kvar: Optional Load A reactive power in kvar (default from DSS file is 1.5).
        load_b_kvar: Optional Load B reactive power in kvar (default from DSS file is 2.0).
        Returns True if compiled and converged.
        """
        dss.Basic.ClearAll()
        if not os.path.exists(self.circuit_file):
            raise FileNotFoundError(f"OpenDSS circuit file not found: {self.circuit_file}")

        cwd = os.getcwd()
        try:
            # Compile circuit
            dss.Command(f'Compile "{self.circuit_file}"')
        finally:
            os.chdir(cwd)

        if initial_soc_percent is not None:
            dss.Command(f"Storage.BSS1.%stored={initial_soc_percent}")
            if reserve_pct is not None:
                dss.Command(f"Storage.BSS1.%reserve={reserve_pct}")
            elif initial_soc_percent <= 20.0:
                # OpenDSS default %reserve=20 blocks discharge when %stored=20.
                # In Data4Cyber testbed, battery discharged from 20% to 0%.
                dss.Command("Storage.BSS1.%reserve=0.0")

        # Apply candidate operating point adjustments if specified
        if slack_pu is not None:
            dss.Command(f"Vsource.Source.pu={slack_pu}")
        if load_a_kvar is not None:
            dss.Command(f"Load.LoadA.kvar={load_a_kvar}")
        if load_b_kvar is not None:
            dss.Command(f"Load.LoadB.kvar={load_b_kvar}")

        dss.Command("Set Mode=Snapshot")
        dss.Command("Solve")

        self._compiled = bool(dss.Solution.Converged())
        return self._compiled

    def set_operating_point(
        self,
        pv_kw: float = None,
        load_a_kw: float = None,
        load_b_kw: float = None,
        bss_state: str = None,
        bss_kw: float = None,
        bss_idling_kw_pct: float = None
    ) -> bool:
        """
        Adjusts DER setpoints and loads, then solves power flow.
        pv_kw: Active power injection in kW for PV (e.g. 20.0).
        load_a_kw: Load A active power in kW (e.g. 7.0).
        load_b_kw: Load B active power in kW (e.g. 8.0).
        bss_state: 'IDLING', 'CHARGING', or 'DISCHARGING'.
        bss_kw: Battery active power setpoint in kW.
        bss_idling_kw_pct: Battery idling parasitic power as % of kWrated (default 1.0 = 200W, set to 0.0 for idealized zero-loss idle).
        """
        if not self._compiled:
            self.build_circuit()

        if bss_idling_kw_pct is not None:
            dss.Command(f"Storage.BSS1.%IdlingkW={bss_idling_kw_pct}")
        if pv_kw is not None:
            dss.Command(f"PVSystem.PV1.pmpp={pv_kw}")
            dss.Command("PVSystem.PV1.irradiance=1.0")
        if load_a_kw is not None:
            dss.Command(f"Load.LoadA.kW={load_a_kw}")
        if load_b_kw is not None:
            dss.Command(f"Load.LoadB.kW={load_b_kw}")
        if bss_state is not None:
            dss.Command(f"Storage.BSS1.State={bss_state.upper()}")
            if bss_kw is not None:
                if bss_state.upper() == "CHARGING":
                    # In OpenDSS Storage, %charge sets charging rate as % of kWrated
                    pct = (bss_kw / 20.0) * 100.0
                    dss.Command(f"Storage.BSS1.%charge={pct}")
                elif bss_state.upper() == "DISCHARGING":
                    pct = (bss_kw / 20.0) * 100.0
                    dss.Command(f"Storage.BSS1.%discharge={pct}")

        dss.Command("Solve")
        return bool(dss.Solution.Converged())

    def step_time_series(self, step_seconds: float = 1.0) -> bool:
        """
        Advances one step in time-series mode.
        """
        dss.Command(f"Set StepSize={step_seconds}s")
        dss.Command("Solve")
        return bool(dss.Solution.Converged())

    def get_bus_voltages(self, bus_name: str) -> Dict[str, float]:
        """
        Extracts 3-phase Line-Neutral and Line-Line RMS voltages for a bus.
        Units: Volts (V).
        """
        dss.Circuit.SetActiveBus(bus_name)
        v_complex = dss.Bus.Voltages()  # [re1, im1, re2, im2, re3, im3]
        if len(v_complex) < 6:
            raise ValueError(f"Bus {bus_name} does not have at least 3 phases.")

        v1 = complex(v_complex[0], v_complex[1])
        v2 = complex(v_complex[2], v_complex[3])
        v3 = complex(v_complex[4], v_complex[5])

        # Line-to-Neutral RMS magnitudes (V)
        uln1 = abs(v1)
        uln2 = abs(v2)
        uln3 = abs(v3)

        # Line-to-Line RMS magnitudes (V)
        ull12 = abs(v1 - v2)
        ull23 = abs(v2 - v3)
        ull31 = abs(v3 - v1)

        return {
            "uln1": float(uln1),
            "uln2": float(uln2),
            "uln3": float(uln3),
            "ull12": float(ull12),
            "ull23": float(ull23),
            "ull31": float(ull31),
        }

    def get_meter_measurements(self, meter_name: str) -> Dict[str, float]:
        """
        Calculates all genuine physical measurements at a designated meter.
        Units:
          - Voltages: Volts (V)
          - Currents: Amperes (A)
          - Powers: Watts (W), var, VA (converts from OpenDSS kW/kvar/kVA by multiplying by 1000)
          - Power factor: -1.0 to 1.0
        """
        if meter_name not in self.METER_LOCATIONS:
            raise KeyError(f"Unknown meter name: {meter_name}")

        cfg = self.METER_LOCATIONS[meter_name]
        element = cfg["element"]
        bus = cfg["bus"]
        terminal = cfg["terminal"]

        # 1. Bus voltages
        v_dict = self.get_bus_voltages(bus)

        # 2. Element powers and currents
        dss.Circuit.SetActiveElement(element)
        all_powers = dss.CktElement.Powers()  # [P1, Q1, P2, Q2, P3, Q3, ...] in kW, kvar
        all_currents = dss.CktElement.Currents()  # [re1, im1, re2, im2, re3, im3, ...] in A

        # Terminal offset dynamically determined by number of conductors per terminal
        num_cond = dss.CktElement.NumConductors()
        offset_p = (terminal - 1) * (num_cond * 2)
        offset_c = (terminal - 1) * (num_cond * 2)

        # Terminal powers per phase in kW, kvar
        p1_kw = all_powers[offset_p + 0]
        q1_kvar = all_powers[offset_p + 1]
        p2_kw = all_powers[offset_p + 2]
        q2_kvar = all_powers[offset_p + 3]
        p3_kw = all_powers[offset_p + 4]
        q3_kvar = all_powers[offset_p + 5]

        # Convert kW / kvar to Watts (W) and var to match Data4Cyber
        # In Data4Cyber benchmark sign convention:
        # At the substation secondary, power exported from the microgrid into the distribution grid
        # is signed NEGATIVE (mean -88W in S0), and power imported from the grid is POSITIVE.
        if meter_name == "Substation-AC-Meter":
            # In OpenDSS Transformer, positive power at terminal 2 means power enters transformer from busbar (exporting).
            # To match Data4Cyber convention (export = negative, import = positive):
            pln1 = -p1_kw * 1000.0
            pln2 = -p2_kw * 1000.0
            pln3 = -p3_kw * 1000.0
            qln1 = -q1_kvar * 1000.0
            qln2 = -q2_kvar * 1000.0
            qln3 = -q3_kvar * 1000.0
        else:
            pln1 = p1_kw * 1000.0
            pln2 = p2_kw * 1000.0
            pln3 = p3_kw * 1000.0
            qln1 = q1_kvar * 1000.0
            qln2 = q2_kvar * 1000.0
            qln3 = q3_kvar * 1000.0

        p_sum3 = pln1 + pln2 + pln3
        q_sum3 = qln1 + qln2 + qln3

        # Apparent power S per phase and 3-phase sum (VA)
        sln1 = (pln1**2 + qln1**2) ** 0.5
        sln2 = (pln2**2 + qln2**2) ** 0.5
        sln3 = (pln3**2 + qln3**2) ** 0.5
        s_sum3 = sln1 + sln2 + sln3

        # Power factors
        cos_phi1 = (pln1 / sln1) if sln1 > 1e-3 else 1.0
        cos_phi2 = (pln2 / sln2) if sln2 > 1e-3 else 1.0
        cos_phi3 = (pln3 / sln3) if sln3 > 1e-3 else 1.0

        # Phase currents in Amperes
        i1_c = complex(all_currents[offset_c + 0], all_currents[offset_c + 1])
        i2_c = complex(all_currents[offset_c + 2], all_currents[offset_c + 3])
        i3_c = complex(all_currents[offset_c + 4], all_currents[offset_c + 5])

        iln1 = abs(i1_c)
        iln2 = abs(i2_c)
        iln3 = abs(i3_c)

        # Vector sum neutral current IN = |I1 + I2 + I3| (Janitza UMG508 register 19018)
        in_vec = i1_c + i2_c + i3_c
        i_sum3 = abs(in_vec)

        # Standard rotation field: +1 for positive (A->B->C)
        phase_seq = 1.0

        return {
            **v_dict,
            "iln1": float(iln1),
            "iln2": float(iln2),
            "iln3": float(iln3),
            "i_sum3": float(i_sum3),
            "pln1": float(pln1),
            "pln2": float(pln2),
            "pln3": float(pln3),
            "p_sum3": float(p_sum3),
            "qln1": float(qln1),
            "qln2": float(qln2),
            "qln3": float(qln3),
            "q_sum3": float(q_sum3),
            "sln1": float(sln1),
            "sln2": float(sln2),
            "sln3": float(sln3),
            "s_sum3": float(s_sum3),
            "cos_phi1": float(cos_phi1),
            "cos_phi2": float(cos_phi2),
            "cos_phi3": float(cos_phi3),
            "phase_seq": float(phase_seq),
        }

    def get_battery_soc(self) -> float:
        """
        Returns true battery State of Charge (%stored) from OpenDSS Storage element.
        """
        dss.Circuit.SetActiveElement("Storage.BSS1")
        soc_str = dss.Properties.Value("%stored")
        return float(soc_str)

    def verify_power_balance(self) -> Tuple[bool, float, Dict[str, float]]:
        """
        Verifies Kirchhoff's Power Law across the microgrid:
        1. Busbar KPL Balance at SubstationBus:
           Sum of active power from transformer secondary (term 2) + all 4 feeder lines (term 1) == 0.0 W.
        2. Microgrid Conservation Balance:
           P_pv + P_bss + P_loadA + P_loadB + P_feeder_losses - P_substation_reported == 0.0 W.
        Returns (is_balanced, residual_watts, power_breakdown_watts).
        """
        meas_sub = self.get_meter_measurements("Substation-AC-Meter")
        meas_pv = self.get_meter_measurements("PV-Inverter-AC-Meter")
        meas_bss = self.get_meter_measurements("BSS-Inverter-AC-Meter")
        meas_la = self.get_meter_measurements("Load-A-AC-Meter")
        meas_lb = self.get_meter_measurements("Load-B-AC-Meter")

        p_sub = meas_sub["p_sum3"]  # Negative when exporting, positive when importing
        p_pv = meas_pv["p_sum3"]    # Negative (generation)
        p_bss = meas_bss["p_sum3"]  # Positive (charging/idle), negative (discharging)
        p_la = meas_la["p_sum3"]    # Positive (load)
        p_lb = meas_lb["p_sum3"]    # Positive (load)

        # Line losses in Watts (OpenDSS CktElement.Losses returns Watts and vars)
        total_line_losses_w = 0.0
        for line in ["Line.FeederPV", "Line.FeederBSS", "Line.FeederLoadA", "Line.FeederLoadB"]:
            dss.Circuit.SetActiveElement(line)
            total_line_losses_w += float(dss.CktElement.Losses()[0])

        # Exact busbar junction balance at SubstationBus
        dss.Circuit.SetActiveElement("Transformer.Substation")
        p_trans = dss.CktElement.Powers()
        # Conductor offset: terminal 2 offset is 8
        p_trans_sec_w = (p_trans[8] + p_trans[10] + p_trans[12]) * 1000.0

        p_feeders_term1_w = 0.0
        for line in ["Line.FeederPV", "Line.FeederBSS", "Line.FeederLoadA", "Line.FeederLoadB"]:
            dss.Circuit.SetActiveElement(line)
            p_l = dss.CktElement.Powers()
            p_feeders_term1_w += (p_l[0] + p_l[2] + p_l[4]) * 1000.0

        busbar_residual_w = abs(p_trans_sec_w + p_feeders_term1_w)

        # Microgrid conservation equation:
        # P_sub (reported) = P_la + P_lb + P_bss + P_pv + P_losses
        expected_p_sub = p_la + p_lb + p_bss + p_pv + total_line_losses_w
        conservation_residual_w = abs(p_sub - expected_p_sub)

        # Both busbar KPL (< 0.01 W) and whole-microgrid conservation (< 1.0 W = 0.005% of 20kW) must be satisfied
        is_balanced = (busbar_residual_w < 0.01) and (conservation_residual_w < 1.0)

        breakdown = {
            "p_substation_reported_w": p_sub,
            "p_pv_inverter_w": p_pv,
            "p_bss_inverter_w": p_bss,
            "p_load_a_w": p_la,
            "p_load_b_w": p_lb,
            "p_feeder_losses_w": total_line_losses_w,
            "busbar_residual_w": busbar_residual_w,
            "conservation_residual_w": conservation_residual_w,
        }
        return is_balanced, max(busbar_residual_w, conservation_residual_w), breakdown
