import numpy as np
from camatis.optimization.stage6_optimization import MultiObjectiveOptimizer

class DecisionOptimizer:

    def __init__(self):
        self.optimizer = MultiObjectiveOptimizer()

    def optimize_actions(self, route_ids, predictions, agent_outputs):
        print("\n" + "="*70)
        print(" OPTIMIZATION LAYER - RESOURCE ALLOCATION")
        print("="*70)
        
        unique_routes = list(set(route_ids))
        
        # Build reduced predictions
        reduced_predictions = {
            "passenger_demand": [],
            "load_factor": [],
            "demand_std": []  
        }

        for route in unique_routes:
            indices = np.where(route_ids == route)[0]
            reduced_predictions["passenger_demand"].append(
                np.mean(predictions['passenger_demand'][indices])
            )
            reduced_predictions["load_factor"].append(
                np.mean(predictions['load_factor'][indices])
            )
            reduced_predictions["demand_std"].append(
                np.mean(predictions['demand_std'][indices])
            )

        reduced_predictions["passenger_demand"] = np.array(reduced_predictions["passenger_demand"])
        reduced_predictions["load_factor"] = np.array(reduced_predictions["load_factor"])
        reduced_predictions["demand_std"] = np.array(reduced_predictions["demand_std"])

        # Run NSGA-III
        results = self.optimizer.optimize(reduced_predictions)
        best = self.optimizer.get_best_solution()
        freq_solution = best['solution']

        optimized_actions = {}

        print("\n📊 Calculating Route Optimizations:")
        
        for i, route in enumerate(unique_routes):
            indices = np.where(route_ids == route)[0]
            actions = agent_outputs.get(route, [])
            
            hourly_demand = reduced_predictions['passenger_demand'][i]
            load_factor = reduced_predictions['load_factor'][i]
            
            # Dynamic bus allocation based on demand
            if hourly_demand > 400:
                buses_to_add = 3
            elif hourly_demand > 250:
                buses_to_add = 2
            elif hourly_demand > 120:
                buses_to_add = 1
            else:
                buses_to_add = 0
            
            # Override if agent flagged for allocation
            if "Allocate Extra Bus" in actions:
                buses_to_add = max(buses_to_add, 2)
            
            route_plan = {
                "frequency_multiplier": freq_solution[i],
                "buses_to_add": buses_to_add,
                "reroute_to": None,
                "reroute_buses_count": 0,  # How many buses to move
                "reroute_reason": None
            }
            
            # REROUTING LOGIC: Move buses from overloaded routes to underloaded ones
            # Condition 1: Route has High Demand Risk AND load factor > 0.8
            # Condition 2: Route has low demand (< 100) - candidate to receive buses from
            if "High Demand Risk" in actions and load_factor > 0.8:
                # Find a route with low demand to take buses FROM
                # Or find a route that can handle overflow
                candidate_routes = []
                for j, candidate in enumerate(unique_routes):
                    if candidate != route:
                        candidate_demand = reduced_predictions['passenger_demand'][j]
                        candidate_load = reduced_predictions['load_factor'][j]
                        
                        # Candidate should have low demand OR low load factor
                        if candidate_demand < 100 or candidate_load < 0.4:
                            candidate_routes.append({
                                "route": candidate,
                                "demand": candidate_demand,
                                "load": candidate_load,
                                "distance": abs(route - candidate) if isinstance(route, (int, float)) else 0
                            })
                
                if candidate_routes:
                    # Sort by load factor (lowest first - best to take from)
                    candidate_routes.sort(key=lambda x: x["load"])
                    best_candidate = candidate_routes[0]
                    
                    # Decide how many buses to reroute
                    # More buses if demand is very high
                    if hourly_demand > 500:
                        buses_to_reroute = 2
                    elif hourly_demand > 300:
                        buses_to_reroute = 1
                    else:
                        buses_to_reroute = 1
                    
                    route_plan["reroute_to"] = best_candidate["route"]
                    route_plan["reroute_buses_count"] = buses_to_reroute
                    route_plan["reroute_reason"] = f"High demand ({hourly_demand:.0f}/hr) - taking {buses_to_reroute} bus(es) from Route {best_candidate['route']}"
                    
                    # Reduce buses on the source route (the one we're taking from)
                    # This will be handled in simulation
                    
            optimized_actions[route] = route_plan
            
            # Print optimization summary (first 30 routes)
            if i < 30:
                status = "🔴 HIGH" if hourly_demand > 300 else "🟡 MEDIUM" if hourly_demand > 150 else "🟢 LOW"
                reroute_info = f" → Reroute to {route_plan['reroute_to']}" if route_plan['reroute_to'] else ""
                print(f"  Route {route}: {status} demand={hourly_demand:.0f} | "
                    f"+{buses_to_add} buses | {route_plan['frequency_multiplier']:.2f}x freq{reroute_info}")
        
        total_buses_added = sum(p["buses_to_add"] for p in optimized_actions.values())
        total_routes_rerouted = sum(1 for p in optimized_actions.values() if p["reroute_to"] is not None)
        total_buses_rerouted = sum(p["reroute_buses_count"] for p in optimized_actions.values() if p["reroute_to"] is not None)
        
        print(f"\n✓ Total additional buses allocated: {total_buses_added}")
        print(f"✓ Routes with rerouting: {total_routes_rerouted}")
        print(f"✓ Buses being rerouted: {total_buses_rerouted}")
        print("="*70)
    
        return optimized_actions

    def find_alternate_route(self, route, reduced_predictions, unique_routes):
        loads = reduced_predictions['load_factor']
        best_score = float("inf")
        best_route = None

        for i, candidate in enumerate(unique_routes):
            if candidate == route:
                continue
            
            distance = abs(route - candidate) if isinstance(route, (int, float)) else 1
            load = loads[i]
            score = 0.6 * distance + 0.4 * load

            if score < best_score:
                best_score = score
                best_route = candidate

        return best_route