import os
from typing import Dict, Any
import base64
from openai import OpenAI
from ..base_model import BaseErgonomicModel


class GPTVisionModel(BaseErgonomicModel):
    """GPT-4o Vision based ergonomic assessment model."""

    def __init__(self):
        """Initialize the GPT-4o Vision model.

        Args:
            api_key (str): OpenAI API key
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI()

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
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "lower_arm": """Analyze the lower arm position and assign a RULA score based on these criteria:
                1 point: 60-100° flexion
                2 points: <60° or >100° flexion
                +1 if arm is working across midline or out to side
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "wrist": """Analyze the wrist position and assign a RULA score based on these criteria:
                1 point: Neutral position
                2 points: 0-15° flexion/extension
                3 points: >15° flexion/extension
                +1 if wrist is bent from midline
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "neck": """Analyze the neck position and assign a RULA score based on these criteria:
                1 point: 0-10° flexion
                2 points: 10-20° flexion
                3 points: >20° flexion
                4 points: In extension
                +1 if neck is twisted
                +1 if neck is side bending
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "trunk": """Analyze the trunk position and assign a RULA score based on these criteria:
                1 point: Sitting well supported at 90°
                2 points: 0-20° flexion
                3 points: 20-60° flexion
                4 points: >60° flexion
                +1 if trunk is twisted
                +1 if trunk is side bending
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "legs": """Analyze the legs position and assign a RULA score based on these criteria:
                1 point: Legs and feet well supported and in balanced position
                2 points: Legs or feet not properly supported or in unbalanced position
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
        }
        return prompts.get(component, "")

    def _get_component_score(self, image_path: str, component: str) -> dict:
        """Get RULA score and analysis for a specific component using GPT-4 Vision."""
        base64_image = self._encode_image(image_path)

        # Simplify the response format to work with older OpenAI library versions
        response_format = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You analyze ergonomic positions and return JSON containing an analysis and score.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._get_component_prompt(component)
                            + '\n\nReturn your response as a JSON with this structure: {"analysis": "your detailed analysis", "score": numeric_score}',
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            response_format=response_format,
            # max_tokens=500,
            temperature=0,
        )

        try:
            import json

            content = response.choices[0].message.content
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"Error getting result for {component}: {e}")
            return {
                "analysis": f"Error analyzing {component}",
                "score": 1,
            }  # Default on error

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image and return RULA component scores with analysis."""
        scores = {}
        analyses = {}
        components = ["upper_arm", "lower_arm", "wrist", "neck", "trunk", "legs"]

        for component in components:
            result = self._get_component_score(image_path, component)
            scores[f"{component}_score"] = result["score"]
            analyses[component] = result["analysis"]

        # Add placeholder for wrist twist score
        scores["wrist_twist_score"] = 1
        analyses["wrist_twist"] = "Not assessed from image"

        return {
            "scores": scores,
            # "analyses": analyses,
            "adjustments": {},  # No adjustments needed as they're included in the GPT scoring
        }

    def get_model_type(self) -> str:
        """Return model type."""
        return "score-based"
