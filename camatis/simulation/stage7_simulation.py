"""
Stage 7: Scenario Simulation Engine
Shows dynamic impact of agent decisions on system state
"""

import simpy
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class RouteState:
    """Tracks state changes for each route"""
    route_id: int
    original_demand: float
    current_demand: float
    original_frequency: float
    current_frequency: float
    original_buses: int
    current_buses: int
    rerouted_from: List[int] = field(default_factory=list)
    rerouted_to: List[int] = field(default_factory=list)
    waiting_passengers: int = 0
    load_factor: float = 0.0


class TransportSimulation:
    """Simulation that shows impact of decisions on system state"""

    def __init__(self, predictions, uncertainty_info, optimized_actions, test_df=None):
        self.predictions = predictions
        self.optimized_actions = optimized_actions
        self.test_df = test_df
        
        # Track state changes
        self.route_states: Dict[int, RouteState] = {}
        self.demand_shifts: List[Dict] = []
        
        # Bus capacity
        self.BUS_CAPACITY = 50
        
    def setup_initial_state(self):
        """Initialize state before any decisions"""
        print("\n" + "="*70)
        print(" INITIAL SYSTEM STATE (Before Decisions)")
        print("="*70)
        
        route_list = list(self.optimized_actions.keys())
        
        for idx, route_id in enumerate(route_list):
            # Get original demand from predictions
            if idx < len(self.predictions.get('passenger_demand', [])):
                demand = float(self.predictions['passenger_demand'][idx])
            else:
                demand = 200.0
            
            demand = max(30, min(800, demand))
            
            # Original frequency (base 4 buses/hour)
            original_freq = 4.0
            original_buses = 1
            
            self.route_states[route_id] = RouteState(
                route_id=route_id,
                original_demand=demand,
                current_demand=demand,
                original_frequency=original_freq,
                current_frequency=original_freq,
                original_buses=original_buses,
                current_buses=original_buses
            )
        
        # Print initial state (first 20 routes)
        print(f"\n{'Route':<8} {'Original Demand':<15} {'Original Freq':<12} {'Original Buses':<12}")
        print("-" * 55)
        for i, (rid, state) in enumerate(list(self.route_states.items())[:20]):
            print(f"{rid:<8} {state.original_demand:<15.0f} {state.original_frequency:<12.1f} {state.original_buses:<12}")
        print(f"... and {len(self.route_states) - 20} more routes")
    
    def apply_frequency_decisions(self):
        """Apply frequency multiplier decisions"""
        print("\n" + "="*70)
        print(" APPLYING FREQUENCY DECISIONS")
        print("="*70)
        
        freq_changes = []
        
        for route_id, plan in self.optimized_actions.items():
            if route_id not in self.route_states:
                continue
            
            multiplier = plan.get("frequency_multiplier", 1.0)
            if multiplier != 1.0:
                old_freq = self.route_states[route_id].current_frequency
                new_freq = old_freq * multiplier
                self.route_states[route_id].current_frequency = new_freq
                freq_changes.append({
                    "route": route_id,
                    "old_freq": old_freq,
                    "new_freq": new_freq,
                    "multiplier": multiplier
                })
        
        # Print frequency changes
        if freq_changes:
            print(f"\n{'Route':<8} {'Old Freq':<10} {'New Freq':<10} {'Multiplier':<10}")
            print("-" * 45)
            for change in freq_changes[:15]:
                print(f"{change['route']:<8} {change['old_freq']:<10.1f} {change['new_freq']:<10.1f} {change['multiplier']:<10.2f}x")
            if len(freq_changes) > 15:
                print(f"... and {len(freq_changes) - 15} more routes")
        else:
            print("  No frequency changes applied")
    
    def apply_bus_allocation_decisions(self):
        """Apply bus allocation decisions"""
        print("\n" + "="*70)
        print(" APPLYING BUS ALLOCATION DECISIONS")
        print("="*70)
        
        bus_changes = []
        
        for route_id, plan in self.optimized_actions.items():
            if route_id not in self.route_states:
                continue
            
            buses_to_add = plan.get("buses_to_add", 0)
            if buses_to_add > 0:
                old_buses = self.route_states[route_id].current_buses
                new_buses = old_buses + buses_to_add
                self.route_states[route_id].current_buses = new_buses
                bus_changes.append({
                    "route": route_id,
                    "old_buses": old_buses,
                    "new_buses": new_buses,
                    "added": buses_to_add
                })
        
        # Print bus allocation changes
        if bus_changes:
            print(f"\n{'Route':<8} {'Old Buses':<10} {'New Buses':<10} {'Added':<8}")
            print("-" * 45)
            for change in bus_changes[:15]:
                print(f"{change['route']:<8} {change['old_buses']:<10} {change['new_buses']:<10} +{change['added']:<7}")
            if len(bus_changes) > 15:
                print(f"... and {len(bus_changes) - 15} more routes")
        else:
            print("  No bus allocation changes")
    
    def apply_rerouting_decisions(self):
        """Apply rerouting decisions - this is where demand shifts happen"""
        print("\n" + "="*70)
        print(" APPLYING REROUTING DECISIONS (DEMAND SHIFTS)")
        print("="*70)
        
        reroute_changes = []
        
        for route_id, plan in self.optimized_actions.items():
            reroute_to = plan.get("reroute_to")
            if reroute_to is not None and reroute_to in self.route_states:
                # Calculate demand shift (30% of demand moves to new route)
                shift_percent = 0.30
                original_demand = self.route_states[route_id].current_demand
                shift_amount = original_demand * shift_percent
                
                # Apply demand shift
                old_from_demand = self.route_states[route_id].current_demand
                old_to_demand = self.route_states[reroute_to].current_demand
                
                self.route_states[route_id].current_demand -= shift_amount
                self.route_states[reroute_to].current_demand += shift_amount
                
                # Track rerouting relationships
                self.route_states[route_id].rerouted_to.append(reroute_to)
                self.route_states[reroute_to].rerouted_from.append(route_id)
                
                self.demand_shifts.append({
                    "from_route": route_id,
                    "to_route": reroute_to,
                    "shift_amount": shift_amount,
                    "shift_percent": shift_percent * 100
                })
                
                reroute_changes.append({
                    "from": route_id,
                    "to": reroute_to,
                    "shift": shift_amount
                })
        
        # Print rerouting changes
        if reroute_changes:
            print(f"\n{'From Route':<12} {'To Route':<12} {'Demand Shifted':<18} {'% Shifted':<10}")
            print("-" * 55)
            for change in reroute_changes[:15]:
                print(f"{change['from']:<12} {change['to']:<12} {change['shift']:<18.0f} 30%")
            if len(reroute_changes) > 15:
                print(f"... and {len(reroute_changes) - 15} more reroutes")
        else:
            print("  No rerouting decisions applied")
    
    def calculate_impact(self):
        """Calculate the impact of all decisions"""
        print("\n" + "="*70)
        print(" IMPACT OF DECISIONS - BEFORE vs AFTER")
        print("="*70)
        
        impacts = []
        
        for route_id, state in self.route_states.items():
            demand_change = state.current_demand - state.original_demand
            demand_change_percent = (demand_change / state.original_demand * 100) if state.original_demand > 0 else 0
            
            freq_change = state.current_frequency - state.original_frequency
            freq_change_percent = (freq_change / state.original_frequency * 100) if state.original_frequency > 0 else 0
            
            bus_change = state.current_buses - state.original_buses
            
            # Calculate capacity (passengers per hour)
            original_capacity = state.original_buses * self.BUS_CAPACITY * state.original_frequency
            new_capacity = state.current_buses * self.BUS_CAPACITY * state.current_frequency
            
            # REALISTIC WAITING CALCULATION
            # If demand exceeds capacity, waiting builds up
            if new_capacity > 0:
                if state.current_demand > new_capacity:
                    # Excess demand per hour, assume 30% of excess accumulates as waiting
                    excess_per_hour = state.current_demand - new_capacity
                    waiting = int(excess_per_hour * 0.3 * 8)  # 8 hour simulation
                else:
                    # Some baseline waiting even with enough capacity
                    waiting = int(state.current_demand * 0.05)
            else:
                waiting = int(state.current_demand * 0.5)
            
            # Calculate load factor (demand / capacity)
            load_factor = min(100, (state.current_demand / new_capacity * 100)) if new_capacity > 0 else 100
            
            state.waiting_passengers = waiting
            state.load_factor = load_factor
            
            if abs(demand_change) > 5 or abs(freq_change) > 0.1 or bus_change != 0:
                impacts.append({
                    "route": route_id,
                    "demand_change": demand_change,
                    "demand_change_percent": demand_change_percent,
                    "freq_change": freq_change,
                    "bus_change": bus_change,
                    "waiting": waiting,
                    "load_factor": load_factor,
                    "original_capacity": original_capacity,
                    "new_capacity": new_capacity
                })
        
        # Print impacts with capacity info
        print(f"\n{'Route':<8} {'Demand Δ':<12} {'Demand %':<10} {'Freq Δ':<10} {'Buses Δ':<10} {'Waiting':<10} {'Load %':<8} {'Capacity':<12}")
        print("-" * 90)
        
        # Sort by absolute demand change
        impacts.sort(key=lambda x: abs(x["demand_change"]), reverse=True)
        
        for impact in impacts[:25]:
            demand_symbol = "↑" if impact["demand_change"] > 0 else "↓" if impact["demand_change"] < 0 else "="
            freq_symbol = "↑" if impact["freq_change"] > 0 else "↓" if impact["freq_change"] < 0 else "="
            bus_symbol = "+" if impact["bus_change"] > 0 else "-" if impact["bus_change"] < 0 else "="
            
            print(f"{impact['route']:<8} "
                f"{demand_symbol}{abs(impact['demand_change']):<11.0f} "
                f"{impact['demand_change_percent']:<+10.1f}% "
                f"{freq_symbol}{abs(impact['freq_change']):<9.1f} "
                f"{bus_symbol}{abs(impact['bus_change']):<9} "
                f"{impact['waiting']:<10} "
                f"{impact['load_factor']:<8.0f}% "
                f"{impact['new_capacity']:<12.0f}")
        
        return impacts
    
    def calculate_efficiency(self):
        """Calculate overall system efficiency after decisions"""
        total_original_demand = sum(s.original_demand for s in self.route_states.values())
        total_new_demand = sum(s.current_demand for s in self.route_states.values())
        total_waiting = sum(s.waiting_passengers for s in self.route_states.values())
        
        # Efficiency: how well demand is being served
        # Lower waiting = higher efficiency
        if total_new_demand > 0:
            efficiency = max(0, 100 - (total_waiting / total_new_demand * 100))
        else:
            efficiency = 100
        
        return {
            "total_original_demand": total_original_demand,
            "total_new_demand": total_new_demand,
            "total_waiting": total_waiting,
            "system_efficiency": round(efficiency, 1),
            "routes_with_demand_shift": len(self.demand_shifts),
            "routes_with_freq_change": sum(1 for s in self.route_states.values() if s.current_frequency != s.original_frequency),
            "routes_with_buses_added": sum(1 for s in self.route_states.values() if s.current_buses > s.original_buses),
            "routes_rerouted": len(self.demand_shifts)
        }
    
    def update_optimized_actions(self):
        """Update the optimized_actions with simulation results"""
        for route_id, state in self.route_states.items():
            if route_id in self.optimized_actions:
                self.optimized_actions[route_id]["demand_after"] = state.current_demand
                self.optimized_actions[route_id]["waiting_passengers"] = state.waiting_passengers
                self.optimized_actions[route_id]["load_factor"] = state.load_factor
                self.optimized_actions[route_id]["demand_change_percent"] = ((state.current_demand - state.original_demand) / state.original_demand * 100) if state.original_demand > 0 else 0
        
        # Also store dynamic demand for final output
        self.dynamic_demand = {rid: s.current_demand for rid, s in self.route_states.items()}
        self.route_queues = {rid: s.waiting_passengers for rid, s in self.route_states.items()}
    
    def run_simulation(self, duration=480):
        """Run the simulation"""
        print("\n" + "🚀"*35)
        print(" CAMATIS DECISION IMPACT SIMULATION")
        print("🚀"*35)
        
        # Step 1: Setup initial state
        self.setup_initial_state()
        
        # Step 2: Apply frequency decisions
        self.apply_frequency_decisions()
        
        # Step 3: Apply bus allocation decisions
        self.apply_bus_allocation_decisions()
        
        # Step 4: Apply rerouting decisions (MOVES BUSES between routes)
        reroutes = self.apply_rerouting_decisions()
        
        # Step 5: Calculate impact
        impacts = self.calculate_impact()
        
        # Step 6: Print rerouting summary
        if reroutes:
            print("\n" + "="*70)
            print(" REROUTING SUMMARY - BUS MOVEMENTS")
            print("="*70)
            total_buses_moved = sum(r["buses_moved"] for r in reroutes)
            print(f"✓ Total buses rerouted: {total_buses_moved}")
            print(f"✓ Routes affected: {len(reroutes)}")
            
            # Show which routes gained/lost capacity
            capacity_gains = {}
            capacity_losses = {}
            for r in reroutes:
                capacity_losses[r["from"]] = capacity_losses.get(r["from"], 0) + abs(r["from_capacity_change"])
                capacity_gains[r["to"]] = capacity_gains.get(r["to"], 0) + r["to_capacity_change"]
            
            if capacity_losses:
                print("\n  Routes that LOST capacity (sent buses away):")
                for rid, loss in list(capacity_losses.items())[:5]:
                    print(f"    Route {rid}: -{loss:.0f} capacity")
            
            if capacity_gains:
                print("\n  Routes that GAINED capacity (received buses):")
                for rid, gain in list(capacity_gains.items())[:5]:
                    print(f"    Route {rid}: +{gain:.0f} capacity")
        
        # Step 7: Update outputs
        self.update_optimized_actions()
        
        return impacts
    def get_results(self):
        """Return results for final output"""
        efficiency = self.calculate_efficiency()
        return {
            "total_original_demand": efficiency['total_original_demand'],
            "total_new_demand": efficiency['total_new_demand'],
            "total_waiting": efficiency['total_waiting'],
            "system_efficiency": efficiency['system_efficiency'],
            "routes_rerouted": efficiency['routes_rerouted'],
            "routes_with_freq_change": efficiency['routes_with_freq_change'],
            "routes_with_buses_added": efficiency['routes_with_buses_added']
        }
    
    def apply_rerouting_decisions(self):
        """Apply rerouting decisions - MOVING BUSES between routes"""
        print("\n" + "="*70)
        print(" APPLYING REROUTING DECISIONS (MOVING BUSES)")
        print("="*70)
        
        reroute_changes = []
        
        for route_id, plan in self.optimized_actions.items():
            reroute_to = plan.get("reroute_to")
            buses_to_move = plan.get("reroute_buses_count", 0)
            
            if reroute_to is not None and buses_to_move > 0:
                if route_id not in self.route_states or reroute_to not in self.route_states:
                    continue
                
                # GET CURRENT BUS COUNTS
                from_buses = self.route_states[route_id].current_buses
                to_buses = self.route_states[reroute_to].current_buses
                
                # GET FREQUENCIES
                from_freq = self.route_states[route_id].current_frequency
                to_freq = self.route_states[reroute_to].current_frequency
                
                # ENSURE WE HAVE ENOUGH BUSES TO MOVE
                buses_to_move = min(buses_to_move, from_buses - 1)  # Leave at least 1 bus
                
                if buses_to_move <= 0:
                    print(f"  ⚠️ Cannot reroute from {route_id} to {reroute_to}: insufficient buses")
                    continue
                
                # CALCULATE CAPACITY BEFORE (capacity = buses * 50 capacity * frequency)
                from_capacity_before = from_buses * self.BUS_CAPACITY * from_freq
                to_capacity_before = to_buses * self.BUS_CAPACITY * to_freq
                
                # MOVE BUSES
                self.route_states[route_id].current_buses -= buses_to_move
                self.route_states[reroute_to].current_buses += buses_to_move
                
                # CALCULATE CAPACITY AFTER
                from_capacity_after = self.route_states[route_id].current_buses * self.BUS_CAPACITY * from_freq
                to_capacity_after = self.route_states[reroute_to].current_buses * self.BUS_CAPACITY * to_freq
                
                # CAPACITY CHANGES
                from_capacity_change = from_capacity_after - from_capacity_before
                to_capacity_change = to_capacity_after - to_capacity_before
                
                # DEMAND SHIFT (15% of demand follows rerouted buses - more realistic)
                demand_shift_percent = 0.15
                from_demand = self.route_states[route_id].current_demand
                demand_shift = from_demand * demand_shift_percent
                
                self.route_states[route_id].current_demand -= demand_shift
                self.route_states[reroute_to].current_demand += demand_shift
                
                # TRACK REROUTING
                self.route_states[route_id].rerouted_to.append({
                    "to_route": reroute_to,
                    "buses_moved": buses_to_move,
                    "demand_shift": demand_shift
                })
                
                reroute_changes.append({
                    "from": route_id,
                    "to": reroute_to,
                    "buses_moved": buses_to_move,
                    "demand_shift": demand_shift,
                    "from_capacity_change": from_capacity_change,
                    "to_capacity_change": to_capacity_change,
                    "from_capacity_before": from_capacity_before,
                    "to_capacity_before": to_capacity_before,
                    "from_capacity_after": from_capacity_after,
                    "to_capacity_after": to_capacity_after
                })
        
        # Print rerouting changes with actual capacity numbers
        if reroute_changes:
            print(f"\n{'From':<8} {'To':<8} {'Buses':<6} {'Demand Shift':<12} {'From Cap':<12} {'To Cap':<12} {'From Δ':<10} {'To Δ':<10}")
            print("-" * 85)
            for change in reroute_changes[:20]:
                print(f"{change['from']:<8} {change['to']:<8} {change['buses_moved']:<6} "
                    f"{change['demand_shift']:<12.0f} "
                    f"{change['from_capacity_before']:<12.0f} "
                    f"{change['to_capacity_before']:<12.0f} "
                    f"{change['from_capacity_change']:<+10.0f} "
                    f"{change['to_capacity_change']:<+10.0f}")
        else:
            print("  No rerouting decisions applied")
        
        return reroute_changes
class ScenarioSimulator:
    """Simulator that shows decision impacts"""
    
    def run(self, predictions, uncertainty_info, optimized_actions, test_df=None, duration=480):
        print("\n" + "="*70)
        print(" RUNNING DECISION IMPACT SIMULATION")
        print("="*70)
        
        sim = TransportSimulation(
            predictions,
            uncertainty_info,
            optimized_actions,
            test_df
        )
        
        sim.run_simulation(duration=duration)
        results = sim.get_results()
        
        return sim, results