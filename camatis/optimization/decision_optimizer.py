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
            "load_factor": []
        }

        for route in unique_routes:

            indices = np.where(route_ids == route)[0]

            reduced_predictions["passenger_demand"].append(
                np.mean(predictions['passenger_demand'][indices])
            )

            reduced_predictions["load_factor"].append(
                np.mean(predictions['load_factor'][indices])
            )

        # Convert to numpy arrays
        reduced_predictions["passenger_demand"] = np.array(reduced_predictions["passenger_demand"])
        reduced_predictions["load_factor"] = np.array(reduced_predictions["load_factor"])

        # Run NSGA-III
        results = self.optimizer.optimize(reduced_predictions)

        best = self.optimizer.get_best_solution()

        freq_solution = best['solution']  # frequency multipliers

        optimized_actions = {}

        for i, route in enumerate(unique_routes):

            actions = agent_outputs.get(route, [])

            route_plan = {
                "frequency_multiplier": freq_solution[i],
                "buses_to_add": 0,
                "reroute_to": None
            }

            # 🚍 Fleet logic
            if "Allocate Extra Bus" in actions:

                demand = reduced_predictions['passenger_demand'][i]
                capacity = 50

                required = int(np.ceil(demand / capacity))
                route_plan["buses_to_add"] = required

            # 🔁 Frequency logic
            if "Increase Frequency" in actions:

                route_plan["frequency_multiplier"] = min(
                    2.0,
                    route_plan["frequency_multiplier"] * 1.2
                )

            # 🛣️ Rerouting logic
            if "High Demand Risk" in actions:

                alt_route = self.find_alternate_route(route, predictions)

                route_plan["reroute_to"] = alt_route

            optimized_actions[route] = route_plan

        return optimized_actions

    def find_alternate_route(self, route, predictions):

        loads = predictions['load_factor']

        for i, l in enumerate(loads):
            if l < 0.5:
                return i

        return None