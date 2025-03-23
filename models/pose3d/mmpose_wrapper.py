import sys
import json
import numpy as np
import tempfile
import os
from mmpose.apis import MMPoseInferencer


def run_inference(image_path, vis_out_dir, pred_out_dir):
    """Run MMPose inference and return the results in a serializable format."""
    try:
        # Initialize the inferencer
        inferencer = MMPoseInferencer(
            pose3d="human3d",
            device="cpu",
        )

        # Run inference
        result_generator = inferencer(
            image_path,
            show=False,
            vis_out_dir=vis_out_dir,
            pred_out_dir=pred_out_dir,
        )

        # Get first result and prepare for serialization
        result = next(result_generator)

        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, list):
                serializable_result[key] = []
                for item in value:
                    if isinstance(item, dict):
                        serialized_item = {}
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                serialized_item[k] = v.tolist()
                            else:
                                serialized_item[k] = v
                        serializable_result[key].append(serialized_item)
                    else:
                        serializable_result[key].append(item)
            else:
                serializable_result[key] = value

        # Return as JSON
        return json.dumps(serializable_result)

    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Expect arguments: image_path, vis_out_dir, pred_out_dir, output_file
    if len(sys.argv) < 5:
        print(json.dumps({"error": "Not enough arguments provided"}))
        sys.exit(1)

    image_path = sys.argv[1]
    vis_out_dir = sys.argv[2]
    pred_out_dir = sys.argv[3]
    output_file = sys.argv[4]

    try:
        result = run_inference(image_path, vis_out_dir, pred_out_dir)

        # Write results to the output file
        with open(output_file, "w") as f:
            f.write(result)

        sys.exit(0)
    except Exception as e:
        error_json = json.dumps({"error": str(e)})
        with open(output_file, "w") as f:
            f.write(error_json)
        sys.exit(1)
