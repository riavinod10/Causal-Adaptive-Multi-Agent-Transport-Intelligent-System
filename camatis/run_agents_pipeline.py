"""
CAMATIS Decision Pipeline
Runs trained models + agents WITHOUT retraining
"""

import os
import sys
import numpy as np

# Ensure the repository root is on the Python path when executed as a script
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from camatis.config import *
from camatis.stage1_data_loader import DataLoader
from camatis.stage3_deep_learning import DeepLearningTrainer
from camatis.stage4_meta_ensemble import MetaEnsemble
from camatis.stage5_uncertainty_anomaly import UncertaintyAnomalyPipeline
from camatis.agents.agent_manager import AgentManager
from camatis.optimization.decision_optimizer import DecisionOptimizer

from camatis.execution.action_engine import TransportState, ExecutionEngine

class CAMATISDecisionPipeline:

    def __init__(self):

        print("\n" + "="*70)
        print(" CAMATIS DECISION SYSTEM")
        print("="*70)

        self.data_loader = DataLoader()
        self.meta_ensemble = MetaEnsemble()
        self.uncertainty_pipeline = UncertaintyAnomalyPipeline()
        self.agent_manager = AgentManager()

    def run(self):

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------
        train_df, test_df = self.data_loader.load_data()
        from camatis.stage2_causal_features import CausalFeatureEngine

        causal_engine = CausalFeatureEngine()

        # Build causal graph
        causal_engine.build_causal_graph()

        # Learn relationships
        causal_engine.learn_causal_relationships(train_df)

        # Generate counterfactual features
        
        test_df = causal_engine.generate_counterfactual_features(test_df)

        X_train, X_test, y_train, y_test, feature_names = \
            self.data_loader.prepare_features(train_df, test_df)

        # --------------------------------------------------
        # Load trained ensemble models
        # --------------------------------------------------
        self.meta_ensemble.load_models()

        # --------------------------------------------------
        # Load trained DL model
        # --------------------------------------------------
        dl_trainer = DeepLearningTrainer(input_dim=X_test.shape[1])
        dl_trainer.load_model()

        dl_predictions_test = dl_trainer.predict(X_test)

        # --------------------------------------------------
        # Meta ensemble predictions
        # --------------------------------------------------
        final_predictions = self.meta_ensemble.predict_meta(
            X_test, dl_predictions_test
        )

        # --------------------------------------------------
        # Uncertainty + anomaly
        # --------------------------------------------------
        uncertainty_results = self.uncertainty_pipeline.process_inference(
    X_train, X_test, dl_trainer
    )

        # --------------------------------------------------
        # Run agents
        # --------------------------------------------------
        decisions = self.agent_manager.process(

            route_ids=test_df["route_id"].values,

            demand_mean=final_predictions['passenger_demand'],
            load_mean=final_predictions['load_factor'],
            utilization=final_predictions['utilization_encoded'],

            demand_std=uncertainty_results['uncertainty_predictions']['demand_std'],
            load_std=uncertainty_results['uncertainty_predictions']['load_std'],

            cls_probs=uncertainty_results['uncertainty_predictions']['cls_probs'],
            high_uncertainty_mask=uncertainty_results['high_uncertainty_mask'],
            anomaly_mask=uncertainty_results['anomaly_results']['is_anomaly']
        )

      
        # -----------------------------------
# OPTIMIZATION LAYER (NEW)
# -----------------------------------

        optimizer = DecisionOptimizer()

        # Convert decisions → route map
        route_actions = {}

        for d in decisions:
            route = d["route_id"]
            actions = d.get("actions", [d.get("action", "No Action")])
            route_actions.setdefault(route, []).extend(actions)

       # 🔥 MERGE ML + UNCERTAINTY
        merged_predictions = {
            **final_predictions,
            **uncertainty_results['uncertainty_predictions'],
            "cf_load_diff": test_df["cf_load_diff"].values,
            "cf_speed_diff": test_df["cf_speed_diff"].values
        }

        optimized = optimizer.optimize_actions(
            route_ids=test_df["route_id"].values,
            predictions=merged_predictions,
            agent_outputs=route_actions
        )

        
   
   
    

        # --------------------------------------------------
# ACTION EXECUTION LAYER (ADD HERE)
# --------------------------------------------------

        #from camatis.execution.action_engine import TransportState, ExecutionEngine

        state = TransportState()
        engine = ExecutionEngine(state)

        print("\nExecuting Actions...\n")

        print("\nExecuting Optimized Actions...\n")

        for route, plan in optimized.items():

            print(f"\nRoute {route} Optimization:")

            freq = plan["frequency_multiplier"]
            buses = plan["buses_to_add"]
            reroute = plan["reroute_to"]

            print(f"→ Frequency Multiplier: {round(freq,2)}x")
            print(f"→ Buses to Add: {buses}")

            if reroute is not None:
                print(f"→ Reroute to: Route {reroute}")
           
            engine.execute_optimized(route, plan)

        print(f"\nAgent decisions generated: {len(decisions)}")
        print("\nSample Decisions:")
        for d in decisions[:10]:
            print(d)

        return decisions


def main():

    np.random.seed(RANDOM_SEED)

    pipeline = CAMATISDecisionPipeline()

    decisions = pipeline.run()

    from collections import defaultdict, Counter

    route_actions = defaultdict(list)

    for d in decisions:
        actions = d.get("actions", [d.get("action", "No Action")])
        route_actions[d["route_id"]].extend(actions)

    route_summary = {}

    for route, acts in route_actions.items():
        if len(acts) > 0:
            route_summary[route] = Counter(acts).most_common(1)[0][0]
        else:
            route_summary[route] = "No Action"

    print("\nRoute-Level Decisions:")
    for r, a in list(route_summary.items())[:10]:
        print(f"Route {r} → {a}")

    # ✅ COUNT DECISIONS PER ROUTE
    action_counts = Counter(route_summary.values())

    print("\nDecision Distribution (Routes):")
    for action, count in action_counts.items():
        percent = (count / len(route_summary)) * 100
        print(f"{action}: {count} routes ({percent:.2f}%)")

    return decisions


if __name__ == "__main__":
    main()