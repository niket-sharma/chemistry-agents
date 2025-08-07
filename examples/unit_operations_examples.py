#!/usr/bin/env python3
"""
Unit Operations Agent Examples

Demonstrates how to use the UnitOperationsAgent for various chemical engineering
unit operation calculations and predictions.

Author: Niket Sharma
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemistry_agents.agents.unit_operations_agent import (
    UnitOperationsAgent, UnitOperationConfig
)
from chemistry_agents.agents.base_agent import AgentConfig


def example_distillation_column():
    """Example: Distillation column design and performance prediction"""
    print("🏭 Distillation Column Example")
    print("=" * 50)
    
    # Create and load the agent
    agent_config = AgentConfig(device="cpu", cpu_optimization=True)
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Define distillation column parameters
    distillation_config = UnitOperationConfig(
        operation_type="distillation",
        temperature=351.15,  # K (78°C - ethanol-water system)
        pressure=101325.0,   # Pa (atmospheric pressure)
        operation_params={
            'alpha': 2.37,        # Relative volatility (ethanol-water at 78°C)
            'xd': 0.89,           # Distillate composition (89% ethanol)
            'xw': 0.02,           # Bottoms composition (2% ethanol)
            'xf': 0.40,           # Feed composition (40% ethanol)
            'viscosity': 0.0008,  # Liquid viscosity (Pa·s)
            'surface_tension': 0.055,  # Surface tension (N/m)
            'packing_type': 'raschig_rings'
        }
    )
    
    # Predict distillation performance
    result = distillation_config_result = agent.predict_single(distillation_config)
    
    print(f"Overall Performance Score: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    
    if 'distillation_results' in result.additional_info:
        dist_results = result.additional_info['distillation_results']
        
        if 'theoretical_stages' in dist_results:
            print(f"Theoretical Stages (Fenske): {dist_results['theoretical_stages']:.1f}")
        
        if 'murphree_efficiency' in dist_results:
            print(f"Murphree Tray Efficiency: {dist_results['murphree_efficiency']:.3f}")
        
        if 'hetp' in dist_results:
            print(f"HETP (Height Equivalent to Theoretical Plate): {dist_results['hetp']:.2f} m")
    
    print()


def example_heat_exchanger():
    """Example: Heat exchanger design and thermal analysis"""
    print("🔥 Heat Exchanger Example")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu")
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Shell-and-tube heat exchanger parameters
    heat_exchanger_config = UnitOperationConfig(
        operation_type="heat_exchanger",
        temperature=323.15,  # K (50°C average temperature)
        pressure=200000.0,   # Pa (2 bar)
        operation_params={
            'reynolds': 8500,              # Reynolds number (turbulent flow)
            'prandtl': 6.8,                # Prandtl number (water)
            'thermal_conductivity': 0.64,  # W/m·K (water at 50°C)
            'diameter': 0.025,             # m (1-inch tube)
            'velocity': 1.8,               # m/s (tube-side velocity)
            'density': 988,                # kg/m³ (water at 50°C)
            'viscosity': 0.00055,          # Pa·s (water at 50°C)
            't_hot_in': 363.15,            # K (90°C hot fluid inlet)
            't_hot_out': 333.15,           # K (60°C hot fluid outlet)
            't_cold_in': 293.15,           # K (20°C cold fluid inlet)
            't_cold_out': 313.15           # K (40°C cold fluid outlet)
        },
        geometry={
            'diameter': 0.025,  # m (tube diameter)
            'length': 4.0,      # m (tube length)
            'shell_diameter': 0.6  # m (shell diameter)
        }
    )
    
    result = agent.predict_single(heat_exchanger_config)
    
    print(f"Heat Exchanger Effectiveness: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    
    if 'heat_exchanger_results' in result.additional_info:
        hx_results = result.additional_info['heat_exchanger_results']
        
        if 'heat_transfer_coefficient' in hx_results:
            h = hx_results['heat_transfer_coefficient']
            print(f"Heat Transfer Coefficient: {h:.1f} W/m²·K")
        
        if 'pressure_drop' in hx_results:
            dp = hx_results['pressure_drop']
            print(f"Pressure Drop: {dp:.0f} Pa ({dp/1000:.1f} kPa)")
        
        if 'lmtd_correction_factor' in hx_results:
            f = hx_results['lmtd_correction_factor']
            print(f"LMTD Correction Factor: {f:.3f}")
    
    print()


def example_chemical_reactor():
    """Example: Chemical reactor design and kinetics analysis"""
    print("⚗️  Chemical Reactor Example")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu")
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # CSTR reactor for A → B reaction
    reactor_config = UnitOperationConfig(
        operation_type="reactor",
        temperature=623.15,  # K (350°C reaction temperature)
        pressure=500000.0,   # Pa (5 bar)
        operation_params={
            'activation_energy': 85000,    # J/mol (typical for organic reactions)
            'pre_exponential': 1.2e9,      # 1/s (frequency factor)
            'concentration': 2.5,          # mol/m³ (reactant A concentration)
            'volume': 1.5,                 # m³ (reactor volume)
            'flow_rate': 0.08,             # m³/s (volumetric flow rate)
            'rate_constant': 0.25,         # 1/s (main reaction rate constant)
            'residence_time': 18.75,       # s (calculated residence time)
            'side_reaction_rate': 0.04     # 1/s (undesired side reaction)
        }
    )
    
    result = agent.predict_single(reactor_config)
    
    print(f"Overall Reactor Performance: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    
    if 'reactor_results' in result.additional_info:
        reactor_results = result.additional_info['reactor_results']
        
        if 'conversion' in reactor_results:
            conversion = reactor_results['conversion']
            print(f"Conversion of A: {conversion:.1%}")
        
        if 'selectivity' in reactor_results:
            selectivity = reactor_results['selectivity']
            print(f"Selectivity to B: {selectivity:.1%}")
        
        if 'residence_time' in reactor_results:
            tau = reactor_results['residence_time']
            print(f"Residence Time: {tau:.1f} seconds ({tau/60:.1f} minutes)")
        
        if 'reaction_rate_constant' in reactor_results:
            k = reactor_results['reaction_rate_constant']
            print(f"Reaction Rate: {k:.2e} mol/m³·s")
    
    print()


def example_separation_process():
    """Example: Separation process (absorption column)"""
    print("🌪️  Absorption Column Example")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu")
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Gas absorption column (CO2 removal from natural gas)
    separation_config = UnitOperationConfig(
        operation_type="separation",
        temperature=298.15,   # K (25°C ambient temperature)
        pressure=2000000.0,   # Pa (20 bar high pressure)
        operation_params={
            'component': 'water',          # Absorbent (water for CO2)
            'diffusivity': 1.6e-9,         # m²/s (CO2 in water)
            'reynolds': 4500,              # Reynolds number (liquid phase)
            'schmidt': 600,                # Schmidt number (CO2-water system)
            'selectivity': 22.0,           # Separation selectivity (CO2 over CH4)
            'capacity': 0.15,              # mol CO2/mol H2O (solubility)
            'gas_flow': 1000,              # kg/h (gas flow rate)
            'liquid_flow': 5000            # kg/h (solvent flow rate)
        }
    )
    
    result = agent.predict_single(separation_config)
    
    print(f"Separation Efficiency: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    
    if 'separation_results' in result.additional_info:
        sep_results = result.additional_info['separation_results']
        
        if 'vapor_liquid_equilibrium' in sep_results:
            vle = sep_results['vapor_liquid_equilibrium']
            if 'k_value' in vle:
                print(f"K-value (y/x): {vle['k_value']:.2f}")
        
        if 'mass_transfer_coefficient' in sep_results:
            kl = sep_results['mass_transfer_coefficient']
            print(f"Liquid-side Mass Transfer Coefficient: {kl:.2e} m/s")
        
        if 'separation_factor' in sep_results:
            alpha_sep = sep_results['separation_factor']
            print(f"Overall Separation Factor: {alpha_sep:.2f}")
    
    print()


def example_fluid_mechanics():
    """Example: Fluid mechanics in process piping"""
    print("💧 Fluid Mechanics Example")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu")
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Process piping system
    fluid_config = UnitOperationConfig(
        operation_type="fluid_mechanics",
        temperature=313.15,  # K (40°C process temperature)
        pressure=300000.0,   # Pa (3 bar gauge pressure)
        operation_params={
            'velocity': 2.5,      # m/s (recommended velocity for liquids)
            'density': 950,       # kg/m³ (process fluid density)
            'viscosity': 0.002,   # Pa·s (process fluid viscosity)
            'diameter': 0.1,      # m (4-inch pipe)
            'length': 150.0,      # m (total pipe length)
            'roughness': 0.00005  # m (commercial steel pipe roughness)
        }
    )
    
    result = agent.predict_single(fluid_config)
    
    print(f"Fluid System Performance: {result.prediction:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    
    if 'fluid_mechanics_results' in result.additional_info:
        fluid_results = result.additional_info['fluid_mechanics_results']
        
        if 'reynolds_number' in fluid_results:
            re = fluid_results['reynolds_number']
            flow_regime = "Turbulent" if re > 2300 else "Laminar"
            print(f"Reynolds Number: {re:.0f} ({flow_regime} flow)")
        
        if 'pressure_drop' in fluid_results:
            dp = fluid_results['pressure_drop']
            print(f"Pressure Drop: {dp:.0f} Pa ({dp/1000:.1f} kPa)")
            print(f"Pressure Drop per 100m: {dp*100/150:.0f} Pa ({dp*100/150/1000:.1f} kPa)")
    
    print()


def batch_processing_example():
    """Example: Batch processing of multiple unit operations"""
    print("🔄 Batch Processing Example")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu", batch_size=4)
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Multiple unit operations for a complete process
    unit_configs = [
        # Feed preheater
        UnitOperationConfig(
            operation_type="heat_exchanger",
            temperature=323.15,
            operation_params={
                'reynolds': 6000, 'prandtl': 7.0,
                'thermal_conductivity': 0.6, 'diameter': 0.02
            }
        ),
        # Reactor
        UnitOperationConfig(
            operation_type="reactor",
            temperature=600.0,
            operation_params={
                'activation_energy': 70000, 'pre_exponential': 5e8,
                'concentration': 1.8, 'volume': 2.0, 'flow_rate': 0.12
            }
        ),
        # Separator
        UnitOperationConfig(
            operation_type="separation",
            temperature=350.0,
            operation_params={
                'component': 'benzene', 'selectivity': 18.0, 'capacity': 0.25
            }
        ),
        # Product cooler
        UnitOperationConfig(
            operation_type="heat_exchanger",
            temperature=313.15,
            operation_params={
                'reynolds': 4500, 'prandtl': 6.5,
                'thermal_conductivity': 0.58, 'diameter': 0.025
            }
        )
    ]
    
    # Batch prediction
    results = agent.predict_batch(unit_configs)
    
    operation_names = ["Feed Preheater", "Reactor", "Separator", "Product Cooler"]
    
    print("Process Unit Performance Summary:")
    print("-" * 35)
    
    overall_performance = 1.0
    for i, (name, result) in enumerate(zip(operation_names, results)):
        print(f"{name:15}: {result.prediction:.3f} (confidence: {result.confidence:.3f})")
        overall_performance *= result.prediction
    
    print("-" * 35)
    print(f"{'Overall Process':15}: {overall_performance:.3f}")
    
    # Identify bottlenecks
    min_performance = min(result.prediction for result in results)
    bottleneck_idx = [i for i, result in enumerate(results) if result.prediction == min_performance][0]
    print(f"Process Bottleneck: {operation_names[bottleneck_idx]} (Performance: {min_performance:.3f})")
    
    print()


def parameter_sensitivity_example():
    """Example: Parameter sensitivity analysis"""
    print("📊 Parameter Sensitivity Analysis")
    print("=" * 50)
    
    agent_config = AgentConfig(device="cpu")
    agent = UnitOperationsAgent(agent_config)
    agent.load_model()
    
    # Base case distillation column
    base_config = UnitOperationConfig(
        operation_type="distillation",
        temperature=351.15,
        pressure=101325.0,
        operation_params={
            'alpha': 2.5,
            'xd': 0.90,
            'xw': 0.05,
            'xf': 0.40
        }
    )
    
    base_result = agent.predict_single(base_config)
    base_performance = base_result.prediction
    
    print(f"Base Case Performance: {base_performance:.3f}")
    print("\nSensitivity Analysis:")
    print("-" * 40)
    
    # Vary relative volatility
    volatilities = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    print("Relative Volatility Effects:")
    for alpha in volatilities:
        config = UnitOperationConfig(
            operation_type="distillation",
            temperature=351.15,
            pressure=101325.0,
            operation_params={
                'alpha': alpha,
                'xd': 0.90,
                'xw': 0.05,
                'xf': 0.40
            }
        )
        
        result = agent.predict_single(config)
        change = ((result.prediction - base_performance) / base_performance) * 100
        print(f"  α = {alpha:.1f}: Performance = {result.prediction:.3f} ({change:+.1f}%)")
    
    print()


def main():
    """Run all unit operations examples"""
    print("🧪 Chemistry Agents - Unit Operations Examples")
    print("=" * 60)
    print("Demonstrating various chemical engineering unit operations")
    print("=" * 60)
    print()
    
    try:
        # Individual unit operation examples
        example_distillation_column()
        example_heat_exchanger()
        example_chemical_reactor()
        example_separation_process()
        example_fluid_mechanics()
        
        # Advanced examples
        batch_processing_example()
        parameter_sensitivity_example()
        
        print("✅ All unit operations examples completed successfully!")
        print("\n💡 Next steps:")
        print("   - Try modifying parameters to see different results")
        print("   - Explore API integration for external model access")
        print("   - Check unit_operations_agent.py for more correlation options")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've installed all required dependencies:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Try running individual examples or check your Python environment")


if __name__ == "__main__":
    main()