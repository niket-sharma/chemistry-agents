"""
Tests for Unit Operations Agent

Tests various chemical engineering unit operations including distillation,
heat exchangers, reactors, separation processes, and fluid mechanics.

Author: Niket Sharma
"""

import pytest
import numpy as np
import math
from unittest.mock import Mock, patch

from chemistry_agents.agents.unit_operations_agent import (
    UnitOperationsAgent, UnitOperationConfig
)
from chemistry_agents.agents.base_agent import AgentConfig, PredictionResult


class TestUnitOperationConfig:
    """Test UnitOperationConfig dataclass"""
    
    def test_config_creation(self):
        """Test basic configuration creation"""
        config = UnitOperationConfig(
            operation_type="distillation",
            temperature=350.0,
            pressure=101325.0
        )
        
        assert config.operation_type == "distillation"
        assert config.temperature == 350.0
        assert config.pressure == 101325.0
        assert config.composition == {}
        assert config.geometry == {}
        assert config.operation_params == {}
    
    def test_config_with_parameters(self):
        """Test configuration with operation parameters"""
        params = {
            'alpha': 2.5,
            'xd': 0.95,
            'xw': 0.05,
            'viscosity': 0.001
        }
        
        config = UnitOperationConfig(
            operation_type="distillation",
            operation_params=params
        )
        
        assert config.operation_params['alpha'] == 2.5
        assert config.operation_params['xd'] == 0.95


class TestUnitOperationsAgent:
    """Test UnitOperationsAgent functionality"""
    
    @pytest.fixture
    def agent(self):
        """Create UnitOperationsAgent instance for testing"""
        config = AgentConfig(device="cpu", cpu_optimization=True)
        agent = UnitOperationsAgent(config)
        return agent
    
    @pytest.fixture
    def loaded_agent(self, agent):
        """Create loaded UnitOperationsAgent instance"""
        agent.load_model()
        return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.config.device == "cpu"
        assert agent.unit_operation_models == {}
        assert not agent.correlations_loaded
        assert not agent.is_loaded
    
    def test_load_model(self, agent):
        """Test model loading"""
        agent.load_model()
        
        assert agent.is_loaded
        assert agent.correlations_loaded
        assert len(agent.antoine_coeffs) > 0
        assert len(agent.critical_properties) > 0
        assert 'distillation' in agent.correlations
        assert 'heat_exchanger' in agent.correlations
    
    def test_supported_operations(self, loaded_agent):
        """Test getting supported operations"""
        operations = loaded_agent.get_supported_operations()
        
        expected_operations = [
            'distillation',
            'heat_exchanger',
            'reactor', 
            'separation',
            'fluid_mechanics'
        ]
        
        for op in expected_operations:
            assert op in operations
    
    def test_operation_parameters(self, loaded_agent):
        """Test getting operation parameters"""
        distillation_params = loaded_agent.get_operation_parameters('distillation')
        
        assert 'alpha' in distillation_params
        assert 'xd' in distillation_params
        assert 'viscosity' in distillation_params
        
        heat_exchanger_params = loaded_agent.get_operation_parameters('heat_exchanger')
        assert 'reynolds' in heat_exchanger_params
        assert 'thermal_conductivity' in heat_exchanger_params


class TestDistillationPredictions:
    """Test distillation column predictions"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_fenske_equation(self, loaded_agent):
        """Test Fenske equation for minimum stages"""
        # Test with realistic distillation parameters
        n_min = loaded_agent._fenske_equation(
            alpha=2.5,  # ethanol-water system
            xd=0.95,    # 95% ethanol in distillate
            xw=0.05,    # 5% ethanol in bottoms  
            xf=0.40     # 40% ethanol in feed
        )
        
        assert isinstance(n_min, float)
        assert n_min > 0
        assert n_min < 50  # Reasonable number of stages
    
    def test_murphree_efficiency(self, loaded_agent):
        """Test Murphree tray efficiency calculation"""
        efficiency = loaded_agent._murphree_efficiency(
            viscosity=0.001,      # Pa·s (water-like)
            surface_tension=0.072  # N/m (water-like)
        )
        
        assert 0.1 <= efficiency <= 1.0
        assert isinstance(efficiency, float)
    
    def test_distillation_prediction(self, loaded_agent):
        """Test complete distillation prediction"""
        config = UnitOperationConfig(
            operation_type="distillation",
            temperature=351.0,  # K
            pressure=101325.0,  # Pa
            operation_params={
                'alpha': 2.37,        # ethanol-water relative volatility
                'xd': 0.89,           # distillate composition
                'xw': 0.02,           # bottoms composition
                'xf': 0.40,           # feed composition
                'viscosity': 0.0008,  # liquid viscosity
                'surface_tension': 0.055,  # surface tension
                'packing_type': 'random'
            }
        )
        
        result = loaded_agent.predict_single(config)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.prediction <= 1
        assert result.confidence > 0
        assert 'distillation_results' in result.additional_info
        
        dist_results = result.additional_info['distillation_results']
        assert 'theoretical_stages' in dist_results
        assert 'murphree_efficiency' in dist_results
        assert 'hetp' in dist_results


class TestHeatExchangerPredictions:
    """Test heat exchanger predictions"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_sieder_tate_correlation(self, loaded_agent):
        """Test Sieder-Tate correlation for Nusselt number"""
        # Turbulent flow conditions
        nu_turbulent = loaded_agent._sieder_tate_correlation(
            re=10000,  # Reynolds number
            pr=7.0     # Prandtl number for water
        )
        
        assert nu_turbulent > 10  # Should be much higher than laminar
        
        # Laminar flow conditions  
        nu_laminar = loaded_agent._sieder_tate_correlation(
            re=1000,   # Reynolds number
            pr=7.0     # Prandtl number
        )
        
        assert nu_laminar == 3.66  # Laminar flow constant
    
    def test_lmtd_correction_factor(self, loaded_agent):
        """Test LMTD correction factor calculation"""
        f_correction = loaded_agent._lmtd_correction_factor(
            t_hot_in=80.0,   # °C
            t_hot_out=60.0,  # °C
            t_cold_in=20.0,  # °C
            t_cold_out=40.0  # °C
        )
        
        assert isinstance(f_correction, float)
        assert f_correction > 0
    
    def test_heat_exchanger_prediction(self, loaded_agent):
        """Test complete heat exchanger prediction"""
        config = UnitOperationConfig(
            operation_type="heat_exchanger",
            temperature=323.15,  # K
            pressure=101325.0,   # Pa
            operation_params={
                'reynolds': 8000,
                'prandtl': 6.8,
                'thermal_conductivity': 0.6,  # W/m·K
                'diameter': 0.025,             # m
                'velocity': 1.5,               # m/s
                'density': 1000,               # kg/m³
                'viscosity': 0.001,            # Pa·s
                't_hot_in': 353.15,            # K
                't_hot_out': 333.15,           # K
                't_cold_in': 293.15,           # K
                't_cold_out': 313.15           # K
            },
            geometry={
                'diameter': 0.025,  # m
                'length': 2.0       # m
            }
        )
        
        result = loaded_agent.predict_single(config)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.prediction <= 1
        assert 'heat_exchanger_results' in result.additional_info
        
        hx_results = result.additional_info['heat_exchanger_results']
        assert 'heat_transfer_coefficient' in hx_results
        assert 'pressure_drop' in hx_results


class TestReactorPredictions:
    """Test reactor performance predictions"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_arrhenius_kinetics(self, loaded_agent):
        """Test Arrhenius kinetics calculation"""
        rate = loaded_agent._arrhenius_kinetics(
            ea=50000,        # J/mol (activation energy)
            a=1e6,           # pre-exponential factor
            temperature=350, # K
            concentration=2.0 # mol/m³
        )
        
        assert isinstance(rate, float)
        assert rate > 0
    
    def test_conversion_selectivity(self, loaded_agent):
        """Test conversion and selectivity calculation"""
        conversion, selectivity = loaded_agent._conversion_selectivity(
            k1=0.1,    # main reaction rate constant
            tau=10,    # residence time
            k2=0.02    # side reaction rate constant
        )
        
        assert 0 <= conversion <= 1
        assert 0 <= selectivity <= 1
        assert isinstance(conversion, float)
        assert isinstance(selectivity, float)
    
    def test_reactor_prediction(self, loaded_agent):
        """Test complete reactor prediction"""
        config = UnitOperationConfig(
            operation_type="reactor",
            temperature=623.15,  # K (350°C)
            pressure=1e5,        # Pa
            operation_params={
                'activation_energy': 75000,    # J/mol
                'pre_exponential': 5e8,        # 1/s
                'concentration': 1.5,          # mol/m³
                'volume': 0.5,                 # m³
                'flow_rate': 0.05,             # m³/s
                'rate_constant': 0.15,         # 1/s
                'residence_time': 10,          # s
                'side_reaction_rate': 0.03     # 1/s
            }
        )
        
        result = loaded_agent.predict_single(config)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.prediction <= 1
        assert 'reactor_results' in result.additional_info
        
        reactor_results = result.additional_info['reactor_results']
        assert 'conversion' in reactor_results
        assert 'selectivity' in reactor_results
        assert 'residence_time' in reactor_results


class TestSeparationProcesses:
    """Test separation process predictions"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_raoults_law_vle(self, loaded_agent):
        """Test vapor-liquid equilibrium using Raoult's law"""
        vle_data = loaded_agent._raoults_law_vle(
            temperature=351.15,  # K
            pressure=101325,     # Pa
            component='ethanol'
        )
        
        assert 'vapor_pressure' in vle_data
        assert 'k_value' in vle_data
        assert vle_data['vapor_pressure'] > 0
        assert vle_data['k_value'] > 0
    
    def test_mass_transfer_coefficient(self, loaded_agent):
        """Test mass transfer coefficient calculation"""
        kl = loaded_agent._mass_transfer_coefficient(
            diffusivity=1e-9,  # m²/s
            re=5000,           # Reynolds number
            sc=600             # Schmidt number
        )
        
        assert isinstance(kl, float)
        assert kl > 0
    
    def test_separation_prediction(self, loaded_agent):
        """Test complete separation prediction"""
        config = UnitOperationConfig(
            operation_type="separation",
            temperature=298.15,   # K
            pressure=101325,      # Pa
            operation_params={
                'component': 'benzene',
                'diffusivity': 8e-10,    # m²/s
                'reynolds': 3000,
                'schmidt': 800,
                'selectivity': 15.0,     # separation selectivity
                'capacity': 0.2          # adsorption capacity
            }
        )
        
        result = loaded_agent.predict_single(config)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.prediction <= 1
        assert 'separation_results' in result.additional_info


class TestFluidMechanics:
    """Test fluid mechanics predictions"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_fluid_mechanics_prediction(self, loaded_agent):
        """Test fluid mechanics calculation"""
        config = UnitOperationConfig(
            operation_type="fluid_mechanics",
            temperature=298.15,  # K
            pressure=101325,     # Pa
            operation_params={
                'velocity': 2.0,     # m/s
                'density': 1000,     # kg/m³
                'viscosity': 0.001,  # Pa·s
                'diameter': 0.05,    # m
                'length': 10.0       # m
            }
        )
        
        result = loaded_agent.predict_single(config)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.prediction <= 1
        assert 'fluid_mechanics_results' in result.additional_info
        
        fluid_results = result.additional_info['fluid_mechanics_results']
        assert 'reynolds_number' in fluid_results
        assert 'pressure_drop' in fluid_results
        
        # Check Reynolds number calculation
        re = fluid_results['reynolds_number']
        expected_re = (1000 * 2.0 * 0.05) / 0.001
        assert abs(re - expected_re) < 1e-6


class TestBatchPredictions:
    """Test batch processing of unit operations"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_predict_batch(self, loaded_agent):
        """Test batch prediction of multiple unit operations"""
        configs = [
            UnitOperationConfig(
                operation_type="distillation",
                operation_params={'alpha': 2.0, 'xd': 0.9, 'xw': 0.1, 'xf': 0.5}
            ),
            UnitOperationConfig(
                operation_type="heat_exchanger", 
                operation_params={
                    'reynolds': 5000, 'prandtl': 7.0, 
                    'thermal_conductivity': 0.6, 'diameter': 0.02
                }
            ),
            UnitOperationConfig(
                operation_type="reactor",
                operation_params={
                    'activation_energy': 60000, 'pre_exponential': 1e7,
                    'concentration': 1.0, 'volume': 1.0, 'flow_rate': 0.1
                }
            )
        ]
        
        results = loaded_agent.predict_batch(configs)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PredictionResult)
            assert 0 <= result.prediction <= 1


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_unsupported_operation_type(self, loaded_agent):
        """Test handling of unsupported operation types"""
        config = UnitOperationConfig(
            operation_type="unsupported_operation",
            operation_params={}
        )
        
        result = loaded_agent.predict_single(config)
        
        assert result.prediction == 0.0
        assert result.confidence == 0.0
        assert 'error' in result.additional_info
    
    def test_prediction_without_loading(self):
        """Test prediction without loading model first"""
        agent = UnitOperationsAgent()
        config = UnitOperationConfig(operation_type="distillation")
        
        with pytest.raises(RuntimeError):
            agent.predict_single(config)
    
    def test_invalid_fenske_parameters(self, loaded_agent):
        """Test Fenske equation with invalid parameters"""
        # Test with alpha <= 1 (invalid)
        result = loaded_agent._fenske_equation(
            alpha=0.8,  # Invalid: should be > 1
            xd=0.95,
            xw=0.05,
            xf=0.5
        )
        
        assert result == float('inf')
        
        # Test with xd <= xw (invalid)
        result = loaded_agent._fenske_equation(
            alpha=2.5,
            xd=0.05,    # Invalid: should be > xw
            xw=0.95,
            xf=0.5
        )
        
        assert result == float('inf')
    
    def test_missing_component_in_vle(self, loaded_agent):
        """Test VLE calculation with unknown component"""
        vle_data = loaded_agent._raoults_law_vle(
            temperature=350,
            pressure=101325,
            component='unknown_component'
        )
        
        assert 'error' in vle_data


class TestPhysicalProperties:
    """Test physical property calculations"""
    
    @pytest.fixture
    def loaded_agent(self):
        config = AgentConfig(device="cpu")
        agent = UnitOperationsAgent(config)
        agent.load_model()
        return agent
    
    def test_antoine_coefficients_loaded(self, loaded_agent):
        """Test that Antoine coefficients are loaded"""
        assert 'water' in loaded_agent.antoine_coeffs
        assert 'ethanol' in loaded_agent.antoine_coeffs
        assert 'benzene' in loaded_agent.antoine_coeffs
        
        # Check coefficient format (A, B, C)
        water_coeffs = loaded_agent.antoine_coeffs['water']
        assert len(water_coeffs) == 3
        assert all(isinstance(coeff, (int, float)) for coeff in water_coeffs)
    
    def test_critical_properties_loaded(self, loaded_agent):
        """Test that critical properties are loaded"""
        assert 'water' in loaded_agent.critical_properties
        assert 'ethanol' in loaded_agent.critical_properties
        
        # Check property format (Tc, Pc, omega)
        water_props = loaded_agent.critical_properties['water']
        assert len(water_props) == 3
        
        tc, pc, omega = water_props
        assert tc > 0      # Critical temperature should be positive
        assert pc > 0      # Critical pressure should be positive
        assert 0 <= omega <= 1  # Acentric factor typically between 0 and 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])