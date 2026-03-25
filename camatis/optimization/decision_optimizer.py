import numpy as np
from camatis.optimization.stage6_optimization import MultiObjectiveOptimizer

class DecisionOptimizer:

    def __init__(self):
        self.optimizer = MultiObjectiveOptimizer()

    def optimize_actions(self, route_ids, predictions, agent_outputs):

        print("\nRunning Optimization Layer...\n")
        # -----------------------------------
# 🔥 REDUCE TO UNIQUE ROUTES
# -----------------------------------

        unique_routes = list(set(route_ids))
        
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

        # Convert to numpy arrays
        reduced_predictions["passenger_demand"] = np.array(reduced_predictions["passenger_demand"])
        reduced_predictions["load_factor"] = np.array(reduced_predictions["load_factor"])
        reduced_predictions["demand_std"] = np.array(reduced_predictions["demand_std"])

        # Run NSGA-III
        results = self.optimizer.optimize(reduced_predictions)

        best = self.optimizer.get_best_solution()

        freq_solution = best['solution']  # frequency multipliers

        optimized_actions = {}

        for i, route in enumerate(unique_routes):
            indices = np.where(route_ids == route)[0]

            actions = agent_outputs.get(route, [])

            route_plan = {
                "frequency_multiplier": freq_solution[i],
                "buses_to_add": 0,
                "reroute_to": None
            }
            action_types = actions

            # 🚍 Fleet logic
            if "Allocate Extra Bus" in action_types:

                expected_load = reduced_predictions['load_factor'][i]

                if expected_load > 0.9:
                    route_plan["buses_to_add"] = 2
                elif expected_load > 0.75 and expected_load<0.9:
                    route_plan["buses_to_add"] = 1

            # 🔁 Frequency logic
            if "Increase Frequency" in action_types:

                cf_impact = np.mean(predictions['cf_speed_diff'][indices])

                if cf_impact > 2:   # big improvement
                    route_plan["frequency_multiplier"] *= 1.3

            # 🛣️ Rerouting logic
            if "High Demand Risk" in action_types:

    # 🔥 Causal feature
                cf_load_reduction = np.mean(predictions['cf_load_diff'][indices])

                # Only reroute if causal improvement exists
                if cf_load_reduction > 0.02:

                    alt_route = self.find_alternate_route(
                        route,
                        reduced_predictions,
                        unique_routes
                    )

                    route_plan["reroute_to"] = alt_route

            optimized_actions[route] = route_plan

        return optimized_actions

    def find_alternate_route(self, route, reduced_predictions, unique_routes):

        loads = reduced_predictions['load_factor']

        best_score = float("inf")
        best_route = None

        for i, candidate in enumerate(unique_routes):

            if candidate == route:
                continue

            # fake distance (since no coords)
            #FIX THIS ADD COORDINATES IN DATASET!!!!!!
            distance = abs(route - candidate)

            load = loads[i]

            # weighted score
            score = 0.6 * distance + 0.4 * load

            if score < best_score:
                best_score = score
                best_route = candidate

        return best_route