import os
import json
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from tabulate import tabulate
import numpy as np
import pandas as pd
from datetime import datetime

from models.pose3d.pose3d_model import Pose3DModel
from models.gpt_vision.gpt_vision_adj_model import GPTVisionAdjModel
from models.gemini_vision.gemini_vision_adj_model import GeminiVisionAdjModel
from models.mppe.mppe_model import Pose3DMPPEModel
from evaluators.rula_evaluator import RULAEvaluator


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def run_with_retry(
    model, image_path: str, max_retries: int = 2
) -> Tuple[Dict[str, Any], float]:
    """Run model with retry logic and measure execution time.

    Args:
        model: The model to run
        image_path: Path to input image
        max_retries: Maximum number of retries

    Returns:
        Tuple of (model_output, execution_time)
    """
    retries = 0
    start_time = time.time()

    while retries <= max_retries:
        try:
            model_output = model.process_image(image_path)
            execution_time = time.time() - start_time
            return model_output, execution_time
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Failed after {max_retries} retries: {str(e)}")
                traceback.print_exc()
                return None, 0
            print(f"Retry {retries}/{max_retries}...")
            time.sleep(1)  # Wait before retry


def format_table(
    results: Dict[str, Dict[str, Any]], timings: Dict[str, float]
) -> List[List]:
    """Format results into a table for printing."""
    table_data = []
    headers = ["Model", "RULA Score", "Time (s)", "Component Scores"]

    for model_name, model_results in results.items():
        if model_results is None or "rula_scores" not in model_results:
            table_data.append([model_name, "Failed", "-", "-"])
            continue

        rula_scores = model_results["rula_scores"]
        component_scores = ", ".join(
            [
                f"{k}: {v}"
                for k, v in rula_scores.items()
                if k
                not in [
                    "final_score",
                    "score_a",
                    "score_b",
                    "table_a_score",
                    "table_b_score",
                ]
            ]
        )

        table_data.append(
            [
                model_name,
                rula_scores["final_score"],
                f"{timings.get(model_name, 0):.2f}",
                component_scores,
            ]
        )

    return [headers] + table_data


def process_dataset(
    dataset_path: str, models: Dict[str, Any], evaluator: RULAEvaluator
) -> Dict[str, Any]:
    """Process a dataset of images and evaluate RULA scores for each model.

    Args:
        dataset_path: Path to dataset directory
        models: Dictionary of models to evaluate
        evaluator: RULA evaluator instance

    Returns:
        Dictionary of results for all images and models
    """
    image_dir = os.path.join(dataset_path, "images")
    results = {}

    # Get all image files
    image_files = [
        f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Found {len(image_files)} images to process")

    for i, image_file in enumerate(sorted(image_files)):
        image_path = os.path.join(image_dir, image_file)
        image_id = os.path.splitext(image_file)[0]
        print(f"\nProcessing image {i + 1}/{len(image_files)}: {image_id}")

        image_results = {}

        for model_name, model in models.items():
            print(f"  Running {model_name}...")
            model_output, execution_time = run_with_retry(model, image_path)

            if model_output is not None:
                # Calculate RULA scores
                rula_scores = evaluator.evaluate(model_output)

                # Store results
                image_results[model_name] = {
                    "model_output": model_output,
                    "rula_scores": rula_scores,
                    "execution_time": execution_time,
                }
                print(
                    f"  ✓ {model_name} completed in {execution_time:.2f}s with RULA score: {rula_scores['final_score']}"
                )
            else:
                image_results[model_name] = None
                print(f"  ✗ {model_name} failed to process the image")

        results[image_id] = image_results

    return results


def calculate_metrics(
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Calculate benchmark metrics for all models across the dataset.

    Args:
        results: Results dictionary

    Returns:
        Dictionary of benchmark metrics
    """
    metrics = {}

    # Get all model names
    model_names = set()
    for image_results in results.values():
        model_names.update(image_results.keys())

    for model_name in model_names:
        model_scores = []
        execution_times = []
        success_count = 0

        for image_id, image_results in results.items():
            if model_name in image_results and image_results[model_name] is not None:
                model_result = image_results[model_name]

                if "rula_scores" in model_result:
                    model_scores.append(model_result["rula_scores"]["final_score"])
                    execution_times.append(model_result["execution_time"])
                    success_count += 1

        # Calculate metrics
        total_images = len(results)
        metrics[model_name] = {
            "success_rate": success_count / total_images if total_images > 0 else 0,
            "avg_rula_score": np.mean(model_scores) if model_scores else None,
            "std_rula_score": np.std(model_scores) if model_scores else None,
            "min_rula_score": np.min(model_scores) if model_scores else None,
            "max_rula_score": np.max(model_scores) if model_scores else None,
            "avg_execution_time": np.mean(execution_times) if execution_times else None,
            "std_execution_time": np.std(execution_times) if execution_times else None,
        }

    return metrics


def main():
    """Main function to process images and evaluate RULA scores."""
    print("\n=== Ergonomic Pose Analysis Benchmark ===\n")

    # Configuration
    input_image = "./inputs/input_old_man.jpeg"
    data_dir = "./data"
    output_dir = "./results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models
    print("Initializing models...")
    models = {
        "Pose3D": Pose3DModel(
            vis_out_dir=f"{output_dir}/pose3d/vis",
            pred_out_dir=f"{output_dir}/pose3d/pred",
        ),
        "GPT4-Vision": GPTVisionAdjModel(),
        "Gemini-Vision": GeminiVisionAdjModel(),
        "MPPE-3D": Pose3DMPPEModel(),
    }

    # Initialize RULA evaluator
    evaluator = RULAEvaluator()

    # Process single image test
    print("\nTest processing a single image...\n")
    single_results = {}
    timings = {}

    for model_name, model in models.items():
        print(f"Running {model_name} on test image...")

        model_output, execution_time = run_with_retry(model, input_image)
        timings[model_name] = execution_time

        if model_output is not None:
            # Calculate RULA scores
            rula_scores = evaluator.evaluate(model_output)

            # Store results
            single_results[model_name] = {
                "model_output": model_output,
                "rula_scores": rula_scores,
            }

            print(f"✓ {model_name} completed in {execution_time:.2f}s")
        else:
            single_results[model_name] = None
            print(f"✗ {model_name} failed to process the test image")

    # Print results table for single image
    table = format_table(single_results, timings)
    print("\nSingle Image Test Results:")
    print(tabulate(table[1:], headers=table[0], tablefmt="grid"))

    # Save single image results
    single_results_with_time = {
        model: {"results": res, "time": timings.get(model, 0)}
        for model, res in single_results.items()
    }
    save_results(single_results_with_time, f"{output_dir}/single_test_{timestamp}.json")

    # Process dataset (all labeled images)
    process_all = input("\nProcess all labeled images? (y/n): ").lower().strip() == "y"

    if process_all:
        print("\nProcessing all labeled images...\n")
        dataset_results = process_dataset(data_dir, models, evaluator)

        # Calculate benchmark metrics
        metrics = calculate_metrics(dataset_results)

        # Print benchmark metrics
        print("\nBenchmark Metrics:")
        metrics_df = pd.DataFrame(metrics).transpose()
        print(metrics_df.to_string())

        # Save results
        save_results(
            {"detailed_results": dataset_results, "benchmark_metrics": metrics},
            f"{output_dir}/benchmark_results_{timestamp}.json",
        )

        # Save metrics as CSV
        metrics_df.to_csv(f"{output_dir}/benchmark_metrics_{timestamp}.csv")

        print(f"\nResults saved to {output_dir}/benchmark_results_{timestamp}.json")
        print(f"Metrics saved to {output_dir}/benchmark_metrics_{timestamp}.csv")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
