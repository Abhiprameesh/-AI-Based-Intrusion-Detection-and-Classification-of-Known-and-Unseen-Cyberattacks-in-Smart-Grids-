"""
Cyber Layer and Controlled Attack Injector for Smart Grid Simulation.
Provides parameterized, reproducible cyber-physical attack scenarios:
1. False Data Injection (FDI / MitM) on AC meter telemetry.
2. Unauthorized Actuator Control Commands (Industroyer PV trip / BSS discharge).
3. Telemetry Manipulation and Dropout (Sensor freeze, packet replay, outage).
4. Coordinated and Gradual Attacks (Unseen stealthy ramp, multi-vector masking).
5. Controlled Benign Physical Disturbances (Solar shading, load switching with Y=0).

Strict Scientific Guardrails:
- Physical state in OpenDSS and reported telemetry are strictly separated.
- Telemetry-only attacks do NOT modify OpenDSS circuit state.
- Control attacks modify physical state, but command vs true state vs reported telemetry are logged separately.
- Ground-truth attack labels and metadata are strictly isolated outside the 150-feature model matrix.
- realtime and freq columns are protected and NEVER modified to simulate attacks.
"""

from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
import copy
import numpy as np


class AttackType(str, Enum):
    FDI_TELEMETRY = "fdi_telemetry"
    UNAUTHORIZED_CONTROL = "unauthorized_control"
    TELEMETRY_DROPOUT = "telemetry_dropout"
    COORDINATED_STEALTH = "coordinated_stealth"
    BENIGN_DISTURBANCE = "benign_disturbance"


class BaseAttack:
    """
    Abstract base class for all cyber-physical attacks and benign controls.
    """

    def __init__(
        self,
        name: str,
        attack_type: AttackType,
        attack_family: str,
        start_step: int,
        end_step: int,
        target_element: str,
        target_signals: List[str],
        is_unseen: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.attack_type = attack_type
        self.attack_family = attack_family
        self.start_step = start_step
        self.end_step = end_step
        self.target_element = target_element
        self.target_signals = target_signals
        self.is_unseen = is_unseen
        self.params = params or {}

    def is_active(self, step: int) -> bool:
        """Returns True if the attack is actively injecting at simulation step."""
        return self.start_step <= step <= self.end_step

    def modify_control_setpoints(
        self, setpoints: Dict[str, Any], step: int
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Modifies actuator commands before being dispatched to OpenDSS.
        By default, non-control attacks do not modify physical setpoints.
        Returns (modified_setpoints, was_modified).
        """
        return setpoints, False

    def modify_reported_telemetry(
        self,
        telemetry: Dict[str, float],
        step: int,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Modifies exported telemetry after OpenDSS power flow is solved.
        By default, non-telemetry attacks do not alter reported measurements.
        Returns (modified_telemetry, was_modified).
        """
        return telemetry, False


class FalseDataInjectionAttack(BaseAttack):
    """
    False Data Injection (FDI / MitM) Attack.
    Simulates malicious tampering of measurement packets across the communication network.
    Physical OpenDSS state is 100% UNCHANGED. Only reported telemetry is corrupted.
    """

    def __init__(
        self,
        name: str = "fdi_bss_meter_scaling",
        target_meter: str = "BSS-Inverter-AC-Meter",
        start_step: int = 50,
        end_step: int = 150,
        scale_factor: float = 0.5,
        bias_w: float = 0.0,
        recompute_apparent_power: bool = True,
        is_unseen: bool = False,
    ):
        signals = [
            f"{target_meter}.p_sum3",
            f"{target_meter}.pln1",
            f"{target_meter}.pln2",
            f"{target_meter}.pln3",
            f"{target_meter}.iln1",
            f"{target_meter}.iln2",
            f"{target_meter}.iln3",
            f"{target_meter}.sln1",
            f"{target_meter}.sln2",
            f"{target_meter}.sln3",
            f"{target_meter}.s_sum3",
        ]
        params = {
            "target_meter": target_meter,
            "scale_factor": scale_factor,
            "bias_w": bias_w,
            "recompute_apparent_power": recompute_apparent_power,
        }
        super().__init__(
            name=name,
            attack_type=AttackType.FDI_TELEMETRY,
            attack_family="fdi_scaling" if scale_factor != 1.0 else "fdi_bias",
            start_step=start_step,
            end_step=end_step,
            target_element=target_meter,
            target_signals=signals,
            is_unseen=is_unseen,
            params=params,
        )
        self.target_meter = target_meter
        self.scale_factor = scale_factor
        self.bias_w = bias_w
        self.recompute_apparent_power = recompute_apparent_power

    def modify_reported_telemetry(
        self,
        telemetry: Dict[str, float],
        step: int,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], bool]:
        if not self.is_active(step):
            return telemetry, False

        corrupted = copy.deepcopy(telemetry)
        m = self.target_meter

        # 1. Scale / bias active powers
        for p_idx in [1, 2, 3]:
            key = f"{m}.pln{p_idx}"
            if key in corrupted:
                corrupted[key] = (corrupted[key] * self.scale_factor) + (self.bias_w / 3.0)

        p_sum_key = f"{m}.p_sum3"
        if p_sum_key in corrupted:
            corrupted[p_sum_key] = (
                corrupted.get(f"{m}.pln1", 0.0)
                + corrupted.get(f"{m}.pln2", 0.0)
                + corrupted.get(f"{m}.pln3", 0.0)
            )

        # 2. Scale currents
        for p_idx in [1, 2, 3]:
            key = f"{m}.iln{p_idx}"
            if key in corrupted:
                corrupted[key] = corrupted[key] * self.scale_factor

        # 3. Maintain coherent apparent powers and power factors if configured
        if self.recompute_apparent_power:
            s_sum = 0.0
            for p_idx in [1, 2, 3]:
                p_val = corrupted.get(f"{m}.pln{p_idx}", 0.0)
                q_val = corrupted.get(f"{m}.qln{p_idx}", 0.0)
                s_val = (p_val**2 + q_val**2) ** 0.5
                corrupted[f"{m}.sln{p_idx}"] = s_val
                s_sum += s_val

                # Power factor
                corrupted[f"{m}.cos_phi{p_idx}"] = (
                    (p_val / s_val) if s_val > 1e-3 else 1.0
                )
            corrupted[f"{m}.s_sum3"] = s_sum

        return corrupted, True


class UnauthorizedControlAttack(BaseAttack):
    """
    Unauthorized Actuator Control Command Attack (Industroyer / Stuxnet style).
    Malicious commands are sent to physical DER controllers (PV inverter or BSS).
    Physical OpenDSS circuit state IS ALTERED. Power flow converges to the new operating point.
    """

    def __init__(
        self,
        name: str = "industroyer_pv_curtailment",
        control_target: str = "PV",  # 'PV' or 'BSS'
        start_step: int = 60,
        end_step: int = 180,
        forced_pv_kw: float = 0.0,
        forced_bss_state: str = "DISCHARGING",
        forced_bss_kw: float = 20.0,
        spoof_reported_soc: bool = False,
        spoofed_soc_value: float = 0.20,
        is_unseen: bool = False,
    ):
        if control_target == "PV":
            signals = [
                "PVSystem.PV1.pmpp",
                "PV-Inverter-AC-Meter.p_sum3",
                "Substation-AC-Meter.p_sum3",
            ]
            family = "unauthorized_pv_trip"
            target_el = "PVSystem.PV1"
        else:
            signals = [
                "Storage.BSS1.State",
                "Storage.BSS1.%discharge",
                "BSS-Inverter-AC-Meter.p_sum3",
                "Battery-SOC-Reference.soc_value",
            ]
            family = "unauthorized_bss_discharge"
            target_el = "Storage.BSS1"

        params = {
            "control_target": control_target,
            "forced_pv_kw": forced_pv_kw,
            "forced_bss_state": forced_bss_state,
            "forced_bss_kw": forced_bss_kw,
            "spoof_reported_soc": spoof_reported_soc,
            "spoofed_soc_value": spoofed_soc_value,
        }
        super().__init__(
            name=name,
            attack_type=AttackType.UNAUTHORIZED_CONTROL,
            attack_family=family,
            start_step=start_step,
            end_step=end_step,
            target_element=target_el,
            target_signals=signals,
            is_unseen=is_unseen,
            params=params,
        )
        self.control_target = control_target
        self.forced_pv_kw = forced_pv_kw
        self.forced_bss_state = forced_bss_state
        self.forced_bss_kw = forced_bss_kw
        self.spoof_reported_soc = spoof_reported_soc
        self.spoofed_soc_value = spoofed_soc_value

    def modify_control_setpoints(
        self, setpoints: Dict[str, Any], step: int
    ) -> Tuple[Dict[str, Any], bool]:
        if not self.is_active(step):
            return setpoints, False

        altered = copy.deepcopy(setpoints)
        if self.control_target == "PV":
            altered["pv_pmpp_kw"] = self.forced_pv_kw
            altered["pv_irradiance_pu"] = 0.0 if self.forced_pv_kw == 0.0 else altered.get("pv_irradiance_pu", 1.0)
        elif self.control_target == "BSS":
            altered["bss_state"] = self.forced_bss_state
            altered["bss_kw"] = self.forced_bss_kw

        return altered, True

    def modify_reported_telemetry(
        self,
        telemetry: Dict[str, float],
        step: int,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], bool]:
        """
        If spoof_reported_soc is True, hides the physical battery drain by clamping
        reported SOC to normal while reference SOC reflects true depletion.
        """
        if not self.is_active(step):
            return telemetry, False

        if self.spoof_reported_soc:
            corrupted = copy.deepcopy(telemetry)
            corrupted["Battery-SOC-Reported.soc_value"] = self.spoofed_soc_value
            return corrupted, True

        return telemetry, False


class TelemetryDropoutAttack(BaseAttack):
    """
    Telemetry Manipulation and Dropout / Denial of Service.
    Simulates communication link loss, meter sensor freezing, or packet replay.
    Physical OpenDSS grid continues operating normally.
    """

    def __init__(
        self,
        name: str = "meter_telemetry_freeze",
        target_meter: str = "Load-B-AC-Meter",
        start_step: int = 50,
        end_step: int = 120,
        dropout_mode: str = "freeze",  # 'freeze' (replay stale value) or 'outage' (zeros)
        is_unseen: bool = False,
    ):
        signals = [f"{target_meter}.{sig}" for sig in ["p_sum3", "q_sum3", "s_sum3", "iln1", "iln2", "iln3"]]
        params = {"target_meter": target_meter, "dropout_mode": dropout_mode}
        super().__init__(
            name=name,
            attack_type=AttackType.TELEMETRY_DROPOUT,
            attack_family=f"telemetry_{dropout_mode}",
            start_step=start_step,
            end_step=end_step,
            target_element=target_meter,
            target_signals=signals,
            is_unseen=is_unseen,
            params=params,
        )
        self.target_meter = target_meter
        self.dropout_mode = dropout_mode
        self._frozen_cache: Optional[Dict[str, float]] = None

    def modify_reported_telemetry(
        self,
        telemetry: Dict[str, float],
        step: int,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], bool]:
        if not self.is_active(step):
            self._frozen_cache = None
            return telemetry, False

        corrupted = copy.deepcopy(telemetry)
        m = self.target_meter

        if self.dropout_mode == "freeze":
            # Cache the first active step's readings or history
            if self._frozen_cache is None:
                self._frozen_cache = {}
                for k, v in telemetry.items():
                    if k.startswith(f"{m}."):
                        # Never freeze realtime or freq
                        if not k.endswith(".realtime") and not k.endswith(".freq"):
                            self._frozen_cache[k] = v

            # Apply cached stale values
            for k, cached_val in self._frozen_cache.items():
                corrupted[k] = cached_val

        elif self.dropout_mode == "outage":
            # Set electrical powers and currents to 0.0 (sensor dead/outage)
            for sig in ["p_sum3", "pln1", "pln2", "pln3", "q_sum3", "qln1", "qln2", "qln3",
                        "s_sum3", "sln1", "sln2", "sln3", "iln1", "iln2", "iln3", "i_sum3"]:
                key = f"{m}.{sig}"
                if key in corrupted:
                    corrupted[key] = 0.0

        return corrupted, True


class UnseenStealthyRampAttack(BaseAttack):
    """
    Reserved Unseen Attack Scenario:
    1. Gradual Stealthy Ramp FDI: Linearly increases/decreases reported measurement to evade threshold alarms.
    2. Coordinated Masking: Commands physical battery discharge while concurrently masking substation telemetry
       so utility meter reports normal net power.
    """

    def __init__(
        self,
        name: str = "unseen_stealthy_ramp_fdi",
        stealth_mode: str = "ramp_fdi",  # 'ramp_fdi' or 'coordinated_masking'
        target_meter: str = "PV-Inverter-AC-Meter",
        start_step: int = 40,
        end_step: int = 200,
        ramp_rate_w_per_sec: float = 25.0,
        max_bias_w: float = 3000.0,
        forced_bss_discharge_kw: float = 10.0,
    ):
        signals = [f"{target_meter}.p_sum3", "Substation-AC-Meter.p_sum3"]
        params = {
            "stealth_mode": stealth_mode,
            "ramp_rate_w_per_sec": ramp_rate_w_per_sec,
            "max_bias_w": max_bias_w,
            "forced_bss_discharge_kw": forced_bss_discharge_kw,
        }
        super().__init__(
            name=name,
            attack_type=AttackType.COORDINATED_STEALTH,
            attack_family="unseen_stealthy_attack",
            start_step=start_step,
            end_step=end_step,
            target_element=target_meter,
            target_signals=signals,
            is_unseen=True,  # Explicitly marked as unseen for generalization testing
            params=params,
        )
        self.stealth_mode = stealth_mode
        self.target_meter = target_meter
        self.ramp_rate = ramp_rate_w_per_sec
        self.max_bias_w = max_bias_w
        self.forced_bss_kw = forced_bss_discharge_kw

    def modify_control_setpoints(
        self, setpoints: Dict[str, Any], step: int
    ) -> Tuple[Dict[str, Any], bool]:
        if not self.is_active(step) or self.stealth_mode != "coordinated_masking":
            return setpoints, False

        # In coordinated masking, covertly discharge the battery
        altered = copy.deepcopy(setpoints)
        altered["bss_state"] = "DISCHARGING"
        altered["bss_kw"] = self.forced_bss_kw
        return altered, True

    def modify_reported_telemetry(
        self,
        telemetry: Dict[str, float],
        step: int,
        history: Optional[List[Dict[str, float]]] = None,
    ) -> Tuple[Dict[str, float], bool]:
        if not self.is_active(step):
            return telemetry, False

        corrupted = copy.deepcopy(telemetry)

        if self.stealth_mode == "ramp_fdi":
            # Linear ramp: bias accumulates each second
            elapsed = step - self.start_step
            bias = min(self.ramp_rate * elapsed, self.max_bias_w)

            # Apply ramp bias to PV active power
            m = self.target_meter
            p_sum_key = f"{m}.p_sum3"
            if p_sum_key in corrupted:
                corrupted[p_sum_key] = corrupted[p_sum_key] + bias
                for p_idx in [1, 2, 3]:
                    corrupted[f"{m}.pln{p_idx}"] = corrupted.get(f"{m}.pln{p_idx}", 0.0) + (bias / 3.0)

        elif self.stealth_mode == "coordinated_masking":
            # Mask substation meter so battery export does not appear on utility meter
            # Battery discharging adds power to microgrid; substation export would increase (more negative).
            # Adversary adds offset to cancel out the battery injection in reported telemetry.
            sub_key = "Substation-AC-Meter.p_sum3"
            if sub_key in corrupted:
                # Add back the discharged Watts so substation appears identical to baseline
                mask_w = self.forced_bss_kw * 1000.0
                corrupted[sub_key] = corrupted[sub_key] + mask_w

        return corrupted, True


class BenignDisturbance(BaseAttack):
    """
    Controlled Benign Physical Disturbance.
    Simulates sudden natural grid events (e.g. sharp cloud shading or large load step).
    Physical state in OpenDSS responds naturally.
    CRITICAL: Ground truth is strictly attack_active = 0.
    Used to verify false positive rates on benign operational variations.
    """

    def __init__(
        self,
        name: str = "benign_cloud_cover_shading",
        disturbance_type: str = "cloud_shading",  # 'cloud_shading' or 'load_step'
        start_step: int = 60,
        end_step: int = 140,
        irradiance_drop_fraction: float = 0.70,  # 70% irradiance dip
        load_step_kw: float = 8.0,               # +8 kW industrial motor start
    ):
        signals = ["PVSystem.PV1.irradiance", "PV-Inverter-AC-Meter.p_sum3"] if disturbance_type == "cloud_shading" else ["Load.LoadA.kW"]
        params = {
            "disturbance_type": disturbance_type,
            "irradiance_drop_fraction": irradiance_drop_fraction,
            "load_step_kw": load_step_kw,
        }
        super().__init__(
            name=name,
            attack_type=AttackType.BENIGN_DISTURBANCE,
            attack_family="benign",
            start_step=start_step,
            end_step=end_step,
            target_element="PVSystem.PV1" if disturbance_type == "cloud_shading" else "Load.LoadA",
            target_signals=signals,
            is_unseen=False,
            params=params,
        )
        self.disturbance_type = disturbance_type
        self.irradiance_drop_fraction = irradiance_drop_fraction
        self.load_step_kw = load_step_kw

    def modify_control_setpoints(
        self, setpoints: Dict[str, Any], step: int
    ) -> Tuple[Dict[str, Any], bool]:
        if not self.is_active(step):
            return setpoints, False

        altered = copy.deepcopy(setpoints)
        if self.disturbance_type == "cloud_shading":
            base_irr = altered.get("pv_irradiance_pu", 1.0)
            altered["pv_irradiance_pu"] = max(0.05, base_irr * (1.0 - self.irradiance_drop_fraction))
        elif self.disturbance_type == "load_step":
            altered["load_a_kw"] = altered.get("load_a_kw", 6.0) + self.load_step_kw

        return altered, True

    def is_attack(self) -> int:
        """Always returns 0 for benign disturbances."""
        return 0
