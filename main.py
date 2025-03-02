import os
import json
from typing import Dict, Any
from models.pose3d.pose3d_model import Pose3DModel
from models.vision_llm.gpt_vision_model import GPTVisionModel
from evaluators.rula_evaluator import RULAEvaluator


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    # Configuration
    input_image = "./others/trial/inputs/input_old_man.jpeg"
    output_dir = "./outputs"
    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )  # Make sure to set this environment variable

    # Initialize models
    models = {
        "pose3d": Pose3DModel(
            vis_out_dir=f"{output_dir}/pose3d/vis",
            pred_out_dir=f"{output_dir}/pose3d/pred",
        ),
        "gpt4_vision": GPTVisionModel(api_key=openai_api_key)
        if openai_api_key
        else None,
    }

    # Initialize RULA evaluator
    evaluator = RULAEvaluator()

    # Process image with each model and evaluate RULA scores
    results = {}
    for model_name, model in models.items():
        if model is None:
            print(f"Skipping {model_name} as it's not configured")
            continue

        print(f"\nProcessing with {model_name}...")

        # Get model output
        model_output = model.process_image(input_image)

        # Calculate RULA scores
        rula_scores = evaluator.evaluate(model_output)

        # Store results
        results[model_name] = {"model_output": model_output, "rula_scores": rula_scores}

        # Print results
        print(f"\n=== {model_name.upper()} Results ===")
        print(f"Final RULA Score: {rula_scores['final_score']}")
        print("\nComponent Scores:")
        for score_name, score in rula_scores.items():
            if score_name not in [
                "final_score",
                "score_a",
                "score_b",
                "table_a_score",
                "table_b_score",
            ]:
                print(f"{score_name}: {score}")

        if model.get_model_type() == "angle-based":
            print("\nDetailed Angles:")
            for angle_name, angle in model_output["angles"].items():
                print(f"{angle_name}: {angle:.1f}°")

    # # Save results
    # save_results(results, f"{output_dir}/results.json")
    # print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
