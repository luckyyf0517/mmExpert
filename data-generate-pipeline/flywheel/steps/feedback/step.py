"""Step 4: classifier-feedback runner."""

from __future__ import annotations

import time
from pathlib import Path

from ...config import FlywheelConfig
from ...logging_utils import get_console, step_panel
from ...path_manager import PathManager
from .analysis import load_classifier_feedback
from .config_update import save_feedback_report, update_config
from .display import ClassifierFeedbackResult, print_summary
from .pipeline import run_classifier_feedback_pipeline

class Step4ClassifierFeedback:
    """Run classifier feedback and update config for the next round."""

    def run(
        self,
        version: str,
        config: FlywheelConfig,
        paths: PathManager,
        *,
        analysis_file: str | Path | None = None,
        test_results_file: str | Path | None = None,
    ) -> ClassifierFeedbackResult:
        """Execute Step 4, generating artifacts automatically unless overridden."""
        analysis_path = Path(analysis_file) if analysis_file is not None else paths.feedback_outputs_dir / "misclassification_analysis.json"
        if test_results_file is not None:
            test_results_path: Path | None = Path(test_results_file)
        elif (paths.feedback_outputs_dir / "test_results.json").exists():
            test_results_path = paths.feedback_outputs_dir / "test_results.json"
        else:
            test_results_path = None

        step_panel(
            "Classifier Feedback",
            subtitle=f"Version: {version}  |  Analysis: {analysis_path.name}",
            step_num=4,
        )

        if not paths.mmwave_dir.exists():
            raise FileNotFoundError(
                f"mmWave directory not found: {paths.mmwave_dir}. Run Step 3 first."
            )
        has_mmwave = any(paths.mmwave_dir.glob("*.npz")) or any(paths.mmwave_dir.glob("*.npy"))
        if not has_mmwave:
            raise FileNotFoundError(
                f"No mmWave outputs found in {paths.mmwave_dir}. Run Step 3 first."
            )

        feedback_dir = paths.feedback_dir
        feedback_dir.mkdir(parents=True, exist_ok=True)
        paths.classifier_outputs_dir.mkdir(parents=True, exist_ok=True)
        paths.feedback_outputs_dir.mkdir(parents=True, exist_ok=True)
        paths.real_data_dir.mkdir(parents=True, exist_ok=True)

        result = ClassifierFeedbackResult()
        start_time = time.time()
        console = get_console()

        if not analysis_path.exists():
            console.print("\n[bold]Phase 1: Running classifier feedback pipeline[/]")
            analysis_path, test_results_path = run_classifier_feedback_pipeline(
                version,
                config,
                paths,
            )

        console.print("\n[bold]Phase 2: Loading classifier feedback artifacts[/]")
        analysis = load_classifier_feedback(analysis_path, test_results_path)
        result.analysis = analysis
        result.analysis_file = str(analysis_path)
        result.test_results_file = str(test_results_path) if test_results_path else ""
        result.overall_accuracy = analysis.overall_accuracy
        result.total_evaluated = analysis.total_evaluated
        result.total_misclassified = analysis.total_misclassified
        result.constraints_added = len(analysis.new_constraints)

        console.print("\n[bold]Phase 3: Updating configuration[/]")
        config = update_config(config, analysis)
        result.config_updated = True

        console.print("\n[bold]Phase 4: Saving normalized report[/]")
        save_feedback_report(analysis, feedback_dir)

        result.elapsed = time.time() - start_time
        print_summary(result, config)
        return result
