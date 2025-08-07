"""
Unit Operations Agent for Chemical Engineering Process Prediction

This agent handles predictions for various chemical engineering unit operations including:
- Distillation columns (tray efficiency, HETP, separation factors)  
- Heat exchangers (heat transfer coefficients, pressure drop, fouling)
- Reactors (conversion, selectivity, residence time, kinetics)
- Separation processes (absorption, extraction, membrane separation)
- Mass transfer operations (drying, crystallization, adsorption)
- Fluid mechanics (pressure drop, flow patterns, mixing)

Author: Niket Sharma
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import math
import warnings

from .base_agent import BaseChemistryAgent, AgentConfig, PredictionResult


@dataclass
class UnitOperationConfig:
    """Configuration for unit operation calculations"""
    operation_type: str  # 'distillation', 'heat_exchanger', 'reactor', etc.
    temperature: float = 298.15  # K
    pressure: float = 101325.0  # Pa
    flow_rate: Optional[float] = None  # kg/s or m3/s
    composition: Optional[Dict[str, float]] = None  # Mole fractions
    geometry: Optional[Dict[str, float]] = None  # Equipment dimensions
    
    # Operation-specific parameters
    operation_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.operation_params is None:
            self.operation_params = {}
        if self.composition is None:
            self.composition = {}
        if self.geometry is None:
            self.geometry = {}


class UnitOperationsAgent(BaseChemistryAgent):
    """
    Agent for predicting chemical engineering unit operation performance
    
    Supports multiple unit operations with both empirical correlations
    and machine learning models for enhanced prediction accuracy.
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(config)
        self.unit_operation_models = {}
        self.correlations_loaded = False
        
        # Physical property databases
        self.antoine_coeffs = {}
        self.critical_properties = {}
        self.transport_properties = {}
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load unit operation models and correlations"""
        try:
            self._load_physical_property_data()
            self._load_empirical_correlations()
            
            if self.config.use_api:
                self._setup_api_model()
            elif model_path:
                self._load_ml_models(model_path)
            else:
                # Use built-in correlations
                self.correlations_loaded = True
                
            self.is_loaded = True
            print(f"✅ Unit Operations Agent loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading Unit Operations Agent: {str(e)}")
            if "internet" in str(e).lower() or "api" in str(e).lower():
                print("💡 Try using offline correlations mode")
            raise
    
    def _load_physical_property_data(self) -> None:
        """Load critical physical property data for calculations"""
        # Antoine equation coefficients (A, B, C) for vapor pressure
        self.antoine_coeffs = {
            'water': (8.07131, 1730.63, -39.724),
            'ethanol': (8.20417, 1642.89, -42.85),
            'methanol': (8.08097, 1582.271, -33.50),
            'benzene': (6.90565, 1211.033, -220.790),
            'toluene': (6.95334, 1343.943, -219.377),
            'acetone': (7.02447, 1161.0, -224),
            'cyclohexane': (6.84498, 1203.526, -222.863),
            'n-hexane': (6.87776, 1171.17, -224.408)
        }
        
        # Critical properties (Tc[K], Pc[Pa], omega)
        self.critical_properties = {
            'water': (647.1, 22064000, 0.3449),
            'ethanol': (513.9, 6137000, 0.6436),
            'methanol': (512.5, 8084000, 0.5625),
            'benzene': (562.05, 4895000, 0.2103),
            'toluene': (591.75, 4108000, 0.2635),
            'acetone': (508.1, 4700000, 0.3071),
            'cyclohexane': (553.8, 4080000, 0.2096),
            'n-hexane': (507.6, 3025000, 0.3013)
        }
    
    def _load_empirical_correlations(self) -> None:
        """Initialize empirical correlations for unit operations"""
        self.correlations = {
            'distillation': {
                'murphree_efficiency': self._murphree_efficiency,
                'fenske_equation': self._fenske_equation,
                'gilliland_correlation': self._gilliland_correlation,
                'hetp_correlation': self._hetp_correlation
            },
            'heat_exchanger': {
                'sieder_tate': self._sieder_tate_correlation,
                'dittus_boelter': self._dittus_boelter_correlation,
                'kern_method': self._kern_method_pressure_drop,
                'lmtd_correction': self._lmtd_correction_factor
            },
            'reactor': {
                'arrhenius_kinetics': self._arrhenius_kinetics,
                'residence_time': self._reactor_residence_time,
                'conversion_selectivity': self._conversion_selectivity,
                'heat_generation': self._reaction_heat_generation
            },
            'separation': {
                'raoults_law': self._raoults_law_vle,
                'henrys_law': self._henrys_law_absorption,
                'mass_transfer_coefficient': self._mass_transfer_coefficient,
                'separation_factor': self._separation_factor
            }
        }
    
    def predict_single(self, 
                      operation_config: UnitOperationConfig) -> PredictionResult:
        """
        Predict unit operation performance for a single configuration
        
        Args:
            operation_config: Configuration for the unit operation
            
        Returns:
            PredictionResult with predicted performance parameters
        """
        if not self.is_loaded:
            raise RuntimeError("Agent not loaded. Call load_model() first.")
        
        try:
            operation_type = operation_config.operation_type.lower()
            
            if operation_type == 'distillation':
                result = self._predict_distillation(operation_config)
            elif operation_type == 'heat_exchanger':
                result = self._predict_heat_exchanger(operation_config)
            elif operation_type == 'reactor':
                result = self._predict_reactor(operation_config)
            elif operation_type == 'separation':
                result = self._predict_separation(operation_config)
            elif operation_type == 'fluid_mechanics':
                result = self._predict_fluid_mechanics(operation_config)
            else:
                raise ValueError(f"Unsupported operation type: {operation_type}")
            
            return result
            
        except Exception as e:
            return PredictionResult(
                smiles="",
                prediction=0.0,
                confidence=0.0,
                additional_info={"error": str(e), "operation": operation_config.operation_type}
            )
    
    def _predict_distillation(self, config: UnitOperationConfig) -> PredictionResult:
        """Predict distillation column performance"""
        params = config.operation_params
        results = {}
        
        # Calculate number of theoretical stages (Fenske equation)
        if 'alpha' in params and 'xd' in params and 'xw' in params:
            alpha = params['alpha']  # Relative volatility
            xd = params['xd']  # Distillate composition
            xw = params['xw']  # Bottoms composition
            xf = params.get('xf', 0.5)  # Feed composition
            
            n_min = self.correlations['distillation']['fenske_equation'](alpha, xd, xw, xf)
            results['theoretical_stages'] = n_min
        
        # Calculate tray efficiency
        if 'viscosity' in params and 'surface_tension' in params:
            efficiency = self.correlations['distillation']['murphree_efficiency'](
                params['viscosity'], params['surface_tension']
            )
            results['murphree_efficiency'] = efficiency
        
        # Calculate HETP for packed columns
        if 'packing_type' in params:
            hetp = self.correlations['distillation']['hetp_correlation'](
                config.temperature, config.pressure, params
            )
            results['hetp'] = hetp
        
        # Overall performance prediction
        overall_performance = np.mean(list(results.values())) if results else 0.5
        
        return PredictionResult(
            smiles="",
            prediction=overall_performance,
            confidence=0.85,
            additional_info={"distillation_results": results, "unit": "dimensionless"}
        )
    
    def _predict_heat_exchanger(self, config: UnitOperationConfig) -> PredictionResult:
        """Predict heat exchanger performance"""
        params = config.operation_params
        results = {}
        
        # Calculate heat transfer coefficient
        if all(k in params for k in ['reynolds', 'prandtl', 'thermal_conductivity']):
            re = params['reynolds']
            pr = params['prandtl']
            k = params['thermal_conductivity']
            d = params.get('diameter', 0.05)  # m
            
            # Sieder-Tate correlation
            nu = self.correlations['heat_exchanger']['sieder_tate'](re, pr)
            h = nu * k / d  # W/m²K
            results['heat_transfer_coefficient'] = h
        
        # Calculate pressure drop
        if 'velocity' in params and 'density' in params and 'viscosity' in params:
            dp = self.correlations['heat_exchanger']['kern_method'](
                params['velocity'], params['density'], params['viscosity'], config.geometry
            )
            results['pressure_drop'] = dp  # Pa
        
        # Calculate LMTD correction factor
        if all(k in params for k in ['t_hot_in', 't_hot_out', 't_cold_in', 't_cold_out']):
            f_correction = self.correlations['heat_exchanger']['lmtd_correction'](
                params['t_hot_in'], params['t_hot_out'], 
                params['t_cold_in'], params['t_cold_out']
            )
            results['lmtd_correction_factor'] = f_correction
        
        # Overall heat transfer effectiveness
        effectiveness = min(results.get('heat_transfer_coefficient', 1000) / 1000, 1.0)
        
        return PredictionResult(
            smiles="",
            prediction=effectiveness,
            confidence=0.80,
            additional_info={"heat_exchanger_results": results, "unit": "effectiveness"}
        )
    
    def _predict_reactor(self, config: UnitOperationConfig) -> PredictionResult:
        """Predict reactor performance"""
        params = config.operation_params
        results = {}
        
        # Calculate reaction rate using Arrhenius equation
        if all(k in params for k in ['activation_energy', 'pre_exponential', 'concentration']):
            k_rate = self.correlations['reactor']['arrhenius_kinetics'](
                params['activation_energy'], params['pre_exponential'], 
                config.temperature, params['concentration']
            )
            results['reaction_rate_constant'] = k_rate
        
        # Calculate residence time
        if 'volume' in params and 'flow_rate' in params:
            tau = self.correlations['reactor']['residence_time'](
                params['volume'], params['flow_rate']
            )
            results['residence_time'] = tau  # seconds
        
        # Calculate conversion and selectivity
        if 'rate_constant' in params and 'residence_time' in params:
            conversion, selectivity = self.correlations['reactor']['conversion_selectivity'](
                params['rate_constant'], params['residence_time'], 
                params.get('side_reaction_rate', 0)
            )
            results['conversion'] = conversion
            results['selectivity'] = selectivity
        
        # Overall reactor performance
        performance = results.get('conversion', 0.5) * results.get('selectivity', 1.0)
        
        return PredictionResult(
            smiles="",
            prediction=performance,
            confidence=0.75,
            additional_info={"reactor_results": results, "unit": "dimensionless"}
        )
    
    def _predict_separation(self, config: UnitOperationConfig) -> PredictionResult:
        """Predict separation process performance"""
        params = config.operation_params
        results = {}
        
        # Vapor-liquid equilibrium using Raoult's law
        if 'component' in params and params['component'] in self.antoine_coeffs:
            vle_data = self.correlations['separation']['raoults_law'](
                config.temperature, config.pressure, params['component']
            )
            results['vapor_liquid_equilibrium'] = vle_data
        
        # Mass transfer coefficient
        if all(k in params for k in ['diffusivity', 'reynolds', 'schmidt']):
            kl = self.correlations['separation']['mass_transfer_coefficient'](
                params['diffusivity'], params['reynolds'], params['schmidt']
            )
            results['mass_transfer_coefficient'] = kl
        
        # Separation factor
        if 'selectivity' in params and 'capacity' in params:
            sep_factor = self.correlations['separation']['separation_factor'](
                params['selectivity'], params['capacity']
            )
            results['separation_factor'] = sep_factor
        
        # Overall separation efficiency
        efficiency = min(results.get('separation_factor', 1.0), 1.0)
        
        return PredictionResult(
            smiles="",
            prediction=efficiency,
            confidence=0.70,
            additional_info={"separation_results": results, "unit": "efficiency"}
        )
    
    def _predict_fluid_mechanics(self, config: UnitOperationConfig) -> PredictionResult:
        """Predict fluid mechanics parameters"""
        params = config.operation_params
        results = {}
        
        # Calculate Reynolds number
        if all(k in params for k in ['velocity', 'density', 'viscosity', 'diameter']):
            re = (params['density'] * params['velocity'] * params['diameter']) / params['viscosity']
            results['reynolds_number'] = re
        
        # Calculate pressure drop (Darcy-Weisbach)
        if 'reynolds_number' in results:
            re = results['reynolds_number']
            # Friction factor for smooth pipes
            if re > 2300:  # Turbulent
                f = 0.316 / (re ** 0.25)
            else:  # Laminar
                f = 64 / re
            
            if all(k in params for k in ['length', 'diameter', 'velocity', 'density']):
                dp = f * (params['length'] / params['diameter']) * \
                     (params['density'] * params['velocity']**2) / 2
                results['pressure_drop'] = dp
        
        # Overall fluid performance metric
        performance = min(1000 / results.get('pressure_drop', 1000), 1.0)
        
        return PredictionResult(
            smiles="",
            prediction=performance,
            confidence=0.85,
            additional_info={"fluid_mechanics_results": results, "unit": "performance"}
        )
    
    # Empirical correlation methods
    def _murphree_efficiency(self, viscosity: float, surface_tension: float) -> float:
        """O'Connell correlation for tray efficiency"""
        # Simplified O'Connell correlation
        efficiency = 0.492 * (viscosity * surface_tension)**(-0.245)
        return min(max(efficiency, 0.1), 1.0)
    
    def _fenske_equation(self, alpha: float, xd: float, xw: float, xf: float) -> float:
        """Fenske equation for minimum number of stages"""
        if alpha <= 1.0 or xd <= xw:
            return float('inf')
        
        numerator = math.log((xd / (1 - xd)) * ((1 - xw) / xw))
        denominator = math.log(alpha)
        return numerator / denominator
    
    def _gilliland_correlation(self, r_min: float, r_actual: float) -> float:
        """Gilliland correlation for actual stages"""
        x = (r_actual - r_min) / (r_actual + 1)
        y = 1 - math.exp(((1 + 54.4*x) / (11 + 117.2*x)) * ((x - 1) / x**0.5))
        return y
    
    def _hetp_correlation(self, temperature: float, pressure: float, params: Dict) -> float:
        """Height Equivalent to Theoretical Plate correlation"""
        # Simplified correlation - actual would depend on packing type
        base_hetp = 0.5  # meters
        temp_factor = (temperature / 298.15)**0.1
        pressure_factor = (101325 / pressure)**0.05
        return base_hetp * temp_factor * pressure_factor
    
    def _sieder_tate_correlation(self, re: float, pr: float) -> float:
        """Sieder-Tate correlation for Nusselt number"""
        if re < 2300:
            return 3.66  # Laminar flow
        return 0.027 * (re**0.8) * (pr**(1/3))
    
    def _dittus_boelter_correlation(self, re: float, pr: float, heating: bool = True) -> float:
        """Dittus-Boelter correlation for Nusselt number"""
        if re < 2300:
            return 3.66
        n = 0.4 if heating else 0.3
        return 0.023 * (re**0.8) * (pr**n)
    
    def _kern_method_pressure_drop(self, velocity: float, density: float, 
                                 viscosity: float, geometry: Dict) -> float:
        """Kern method for shell-side pressure drop"""
        # Simplified pressure drop calculation
        re = density * velocity * geometry.get('diameter', 0.025) / viscosity
        f = 0.316 / (re**0.25) if re > 2300 else 64/re
        return f * geometry.get('length', 1.0) / geometry.get('diameter', 0.025) * \
               density * velocity**2 / 2
    
    def _lmtd_correction_factor(self, t_hot_in: float, t_hot_out: float,
                              t_cold_in: float, t_cold_out: float) -> float:
        """LMTD correction factor for heat exchangers"""
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in
        
        if abs(dt1 - dt2) < 1e-6:
            return 1.0
            
        lmtd = (dt1 - dt2) / math.log(dt1 / dt2)
        return lmtd / ((dt1 + dt2) / 2)  # Simplified correction factor
    
    def _arrhenius_kinetics(self, ea: float, a: float, temperature: float, 
                          concentration: float) -> float:
        """Arrhenius equation for reaction rate"""
        r = 8.314  # J/mol·K
        k = a * math.exp(-ea / (r * temperature))
        return k * concentration
    
    def _reactor_residence_time(self, volume: float, flow_rate: float) -> float:
        """Calculate residence time in reactor"""
        return volume / flow_rate
    
    def _conversion_selectivity(self, k1: float, tau: float, k2: float = 0) -> tuple:
        """Calculate conversion and selectivity"""
        conversion = 1 - math.exp(-k1 * tau)
        selectivity = k1 / (k1 + k2) if k2 > 0 else 1.0
        return conversion, selectivity
    
    def _reaction_heat_generation(self, rate: float, heat_of_reaction: float) -> float:
        """Calculate heat generation in reactor"""
        return rate * heat_of_reaction
    
    def _raoults_law_vle(self, temperature: float, pressure: float, component: str) -> Dict:
        """Vapor-liquid equilibrium using Raoult's law"""
        if component not in self.antoine_coeffs:
            return {"error": f"Component {component} not in database"}
        
        a, b, c = self.antoine_coeffs[component]
        # Antoine equation: log10(P_sat) = A - B/(C + T)
        p_sat = 10**(a - b/(c + temperature - 273.15)) * 1000  # Convert to Pa
        
        return {
            "vapor_pressure": p_sat,
            "activity_coefficient": 1.0,  # Ideal solution
            "k_value": p_sat / pressure
        }
    
    def _henrys_law_absorption(self, temperature: float, component: str) -> float:
        """Henry's law constant for absorption"""
        # Simplified - actual would require Henry's constant database
        base_henry = 1e5  # Pa·m³/mol
        temp_correction = math.exp(2000 * (1/temperature - 1/298.15))
        return base_henry * temp_correction
    
    def _mass_transfer_coefficient(self, diffusivity: float, re: float, sc: float) -> float:
        """Mass transfer coefficient correlation"""
        sh = 0.023 * (re**0.8) * (sc**0.33)  # Sherwood number
        return sh * diffusivity / 0.05  # Assuming characteristic length of 5 cm
    
    def _separation_factor(self, selectivity: float, capacity: float) -> float:
        """Overall separation factor"""
        return selectivity * capacity / (1 + capacity)

    def predict_batch(self, operation_configs: List[UnitOperationConfig]) -> List[PredictionResult]:
        """Predict multiple unit operations in batch"""
        results = []
        for config in operation_configs:
            result = self.predict_single(config)
            results.append(result)
        return results

    def get_supported_operations(self) -> List[str]:
        """Get list of supported unit operations"""
        return [
            'distillation',
            'heat_exchanger', 
            'reactor',
            'separation',
            'fluid_mechanics'
        ]

    def get_operation_parameters(self, operation_type: str) -> Dict[str, str]:
        """Get required parameters for a specific operation type"""
        param_docs = {
            'distillation': {
                'alpha': 'Relative volatility',
                'xd': 'Distillate composition (mole fraction)',
                'xw': 'Bottoms composition (mole fraction)', 
                'xf': 'Feed composition (mole fraction)',
                'viscosity': 'Liquid viscosity (Pa·s)',
                'surface_tension': 'Surface tension (N/m)'
            },
            'heat_exchanger': {
                'reynolds': 'Reynolds number',
                'prandtl': 'Prandtl number',
                'thermal_conductivity': 'Thermal conductivity (W/m·K)',
                'diameter': 'Tube diameter (m)',
                'velocity': 'Fluid velocity (m/s)',
                'density': 'Fluid density (kg/m³)',
                'viscosity': 'Fluid viscosity (Pa·s)',
                't_hot_in': 'Hot fluid inlet temperature (K)',
                't_hot_out': 'Hot fluid outlet temperature (K)',
                't_cold_in': 'Cold fluid inlet temperature (K)',
                't_cold_out': 'Cold fluid outlet temperature (K)'
            },
            'reactor': {
                'activation_energy': 'Activation energy (J/mol)',
                'pre_exponential': 'Pre-exponential factor (1/s)',
                'concentration': 'Reactant concentration (mol/m³)',
                'volume': 'Reactor volume (m³)',
                'flow_rate': 'Flow rate (m³/s)',
                'side_reaction_rate': 'Side reaction rate constant (1/s)'
            },
            'separation': {
                'component': 'Component name (e.g., "water", "ethanol")',
                'diffusivity': 'Diffusivity (m²/s)',
                'reynolds': 'Reynolds number',
                'schmidt': 'Schmidt number',
                'selectivity': 'Separation selectivity',
                'capacity': 'Adsorption capacity (mol/kg)'
            },
            'fluid_mechanics': {
                'velocity': 'Fluid velocity (m/s)',
                'density': 'Fluid density (kg/m³)',
                'viscosity': 'Fluid viscosity (Pa·s)',
                'diameter': 'Pipe diameter (m)',
                'length': 'Pipe length (m)'
            }
        }
        
        return param_docs.get(operation_type, {})