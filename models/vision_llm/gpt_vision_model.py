import os
from typing import Dict, Any
import base64
import requests
from ..base_model import BaseErgonomicModel


class GPTVisionModel(BaseErgonomicModel):
    """GPT-4 Vision based ergonomic assessment model."""

    def __init__(self, api_key: str):
        """Initialize the GPT-4 Vision model.

        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_component_prompt(self, component: str) -> str:
        """Get the specific prompt for each RULA component."""
        prompts = {
            "upper_arm": """Analyze the upper arm position in this image and assign a RULA score based on these criteria:
                1 point: 20° extension to 20° flexion
                2 points: >20° extension or 20-45° flexion
                3 points: 45-90° flexion
                4 points: >90° flexion
                +1 if shoulder is raised
                +1 if upper arm is abducted
                -1 if arm is supported or person is leaning
                
                Return only the numeric score.""",
            "lower_arm": """Analyze the lower arm position and assign a RULA score based on these criteria:
                1 point: 60-100° flexion
                2 points: <60° or >100° flexion
                +1 if arm is working across midline or out to side
                
                Return only the numeric score.""",
            "wrist": """Analyze the wrist position and assign a RULA score based on these criteria:
                1 point: Neutral position
                2 points: 0-15° flexion/extension
                3 points: >15° flexion/extension
                +1 if wrist is bent from midline
                
                Return only the numeric score.""",
            "neck": """Analyze the neck position and assign a RULA score based on these criteria:
                1 point: 0-10° flexion
                2 points: 10-20° flexion
                3 points: >20° flexion
                4 points: In extension
                +1 if neck is twisted
                +1 if neck is side bending
                
                Return only the numeric score.""",
            "trunk": """Analyze the trunk position and assign a RULA score based on these criteria:
                1 point: Sitting well supported at 90°
                2 points: 0-20° flexion
                3 points: 20-60° flexion
                4 points: >60° flexion
                +1 if trunk is twisted
                +1 if trunk is side bending
                
                Return only the numeric score.""",
            "legs": """Analyze the legs position and assign a RULA score based on these criteria:
                1 point: Legs and feet well supported and in balanced position
                2 points: Legs or feet not properly supported or in unbalanced position
                
                Return only the numeric score.""",
        }
        return prompts.get(component, "")

    def _get_component_score(self, image_path: str, component: str) -> int:
        """Get RULA score for a specific component using GPT-4 Vision."""
        base64_image = self._encode_image(image_path)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._get_component_prompt(component)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 10,
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        if response.status_code == 200:
            try:
                score = int(response.json()["choices"][0]["message"]["content"].strip())
                return score
            except (ValueError, KeyError):
                return 1  # Default to minimum score on error
        return 1

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return direct RULA component scores."""
        scores = {}
        components = ["upper_arm", "lower_arm", "wrist", "neck", "trunk", "legs"]

        for component in components:
            scores[f"{component}_score"] = self._get_component_score(
                image_path, component
            )

        # Add placeholder for wrist twist score as it's part of RULA but hard to assess from images
        scores["wrist_twist_score"] = 1

        return {
            "scores": scores,
            "adjustments": {},  # No adjustments needed as they're included in the GPT scoring
        }

    def get_model_type(self) -> str:
        """Return model type."""
        return "score-based"
