"""
Stage 7: Scenario Simulation Engine
Simulates transport scenarios using uncertainty-aware predictions
"""

import simpy
import numpy as np
from camatis.config import *

class Bus:
    """Bus entity in simulation"""
    def __init__(self, bus_id, capacity, route):
        self.bus_id = bus_id
        self.capacity = capacity
        self.route = route
        self.passengers = 0
        self.trips_completed = 0
        
class Route:
    """Route entity"""
    def __init__(self, route_id, frequency, demand_predictor):
        self.route_id = route_id
        self.frequency = frequency  # buses per hour
        self.demand_predictor = demand_predictor
        self.total_passengers = 0
        self.waiting_passengers = 0

class TransportSimulation:
    """Transport system simulation"""
    
    def __init__(self, predictions, uncertainty_info):
        self.env = simpy.Environment()
        self.predictions = predictions
        self.uncertainty_info = uncertainty_info
        self.buses = []
        self.routes = {}
        self.metrics = {
            'total_passengers': 0,
            'total_waiting_time': 0,
            'overload_events': 0,
            'underutilization_events': 0
        }
        
    def setup_routes(self, n_routes=10):
        """Initialize routes"""
        for i in range(n_routes):
            route = Route(
                route_id=i,
                frequency=4,  # 4 buses per hour
                demand_predictor=lambda: np.random.normal(50, 10)
            )
            self.routes[i] = route
    
    def setup_buses(self, n_buses=20):
        """Initialize bus fleet"""
        for i in range(n_buses):
            bus = Bus(
                bus_id=f"BUS_{i:02d}",
                capacity=50,
                route=i % len(self.routes)
            )
            self.buses.append(bus)
    
    def passenger_arrival(self, route):
        """Passenger arrival process"""
        while True:
            # Use uncertainty-aware demand prediction
            base_demand = self.predictions['passenger_demand'][route.route_id % len(self.predictions['passenger_demand'])]
            uncertainty = self.uncertainty_info['confidence_intervals']['demand_std'][route.route_id % len(self.predictions['passenger_demand'])]
            
            # Sample from distribution
            demand = np.random.normal(base_demand, uncertainty)
            demand = max(0, demand)  # Non-negative
            
            route.waiting_passengers += demand
            self.metrics['total_passengers'] += demand
            
            yield self.env.timeout(60 / route.frequency)  # Inter-arrival time
    
    def bus_operation(self, bus):
        """Bus operation process"""
        route = self.routes[bus.route]
        
        while True:
            # Pick up passengers
            passengers_to_board = min(route.waiting_passengers, bus.capacity)
            bus.passengers = passengers_to_board
            route.waiting_passengers -= passengers_to_board
            
            # Check overload/underutilization
            utilization = bus.passengers / bus.capacity
            if utilization > 0.9:
                self.metrics['overload_events'] += 1
            elif utilization < 0.3:
                self.metrics['underutilization_events'] += 1
            
            # Travel time
            yield self.env.timeout(30)  # 30 minutes trip
            
            # Drop off passengers
            bus.passengers = 0
            bus.trips_completed += 1
            
            # Return time
            yield self.env.timeout(10)  # 10 minutes return
    
    def run_simulation(self, duration=SIMULATION_DURATION * 60):
        """Run simulation"""
        print(f"Running simulation for {duration} minutes...")
        
        # Setup
        self.setup_routes()
        self.setup_buses()
        
        # Start processes
        for route in self.routes.values():
            self.env.process(self.passenger_arrival(route))
        
        for bus in self.buses:
            self.env.process(self.bus_operation(bus))
        
        # Run
        self.env.run(until=duration)
        
        print("Simulation completed!")
        return self.metrics
    
    def get_results(self):
        """Get simulation results"""
        return {
            'total_passengers': self.metrics['total_passengers'],
            'avg_waiting_time': self.metrics['total_waiting_time'] / max(self.metrics['total_passengers'], 1),
            'overload_events': self.metrics['overload_events'],
            'underutilization_events': self.metrics['underutilization_events'],
            'total_trips': sum(bus.trips_completed for bus in self.buses)
        }

class ScenarioSimulator:
    """Simulate multiple scenarios"""
    
    def __init__(self):
        self.scenarios = SIMULATION_SCENARIOS
        self.results = {}
    
    def simulate_scenario(self, scenario_name, predictions, uncertainty_info):
        """Simulate a specific scenario"""
        print(f"\n=== Simulating: {scenario_name} ===")
        
        # Modify predictions based on scenario
        modified_predictions = self._apply_scenario_modifications(
            scenario_name, predictions
        )
        
        # Run simulation
        sim = TransportSimulation(modified_predictions, uncertainty_info)
        metrics = sim.run_simulation()
        results = sim.get_results()
        
        self.results[scenario_name] = results
        
        return results
    
    def _apply_scenario_modifications(self, scenario_name, predictions):
        """Apply scenario-specific modifications"""
        modified = predictions.copy()
        
        if scenario_name == 'festival_surge':
            # 2x demand increase
            modified['passenger_demand'] = predictions['passenger_demand'] * 2.0
        
        elif scenario_name == 'congestion_spike':
            # Reduced speed, increased congestion
            pass  # Already in predictions
        
        elif scenario_name == 'fleet_breakdown':
            # Reduced capacity
            modified['load_factor'] = predictions['load_factor'] * 1.3
        
        return modified
    
    def run_all_scenarios(self, predictions, uncertainty_info):
        """Run all predefined scenarios"""
        print("\n" + "="*50)
        print("SCENARIO SIMULATION ENGINE")
        print("="*50)
        
        for scenario in self.scenarios:
            self.simulate_scenario(scenario, predictions, uncertainty_info)
        
        return self.results
    
    def compare_scenarios(self):
        """Compare results across scenarios"""
        print("\n=== Scenario Comparison ===")
        for scenario, results in self.results.items():
            print(f"\n{scenario}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.2f}")
