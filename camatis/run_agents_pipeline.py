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
            X_test, dl_trainer
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

        print(f"\nAgent decisions generated: {len(decisions)}")
        print("\nSample Decisions:")
        for d in decisions[:10]:
            print(d)

        return decisions


def main():

    np.random.seed(RANDOM_SEED)

    pipeline = CAMATISDecisionPipeline()

    decisions = pipeline.run()

    return decisions


if __name__ == "__main__":
    main()