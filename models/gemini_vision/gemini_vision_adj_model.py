import os
from typing import Dict, Any
import base64
import PIL.Image
import google.generativeai as genai
from ..base_model import BaseErgonomicModel


class GeminiVisionAdjModel(BaseErgonomicModel):
    """Gemini Vision based ergonomic assessment model."""

    def __init__(self):
        """Initialize the Gemini Vision model.

        Args:
            api_key (str): Google API key
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _get_component_prompt(self, component: str) -> str:
        """Get the specific prompt for each RULA component."""
        base_prompts = {
            "upper_arm": """Analyze the upper arm position in this image and assign a RULA base score based on these criteria:
                1 point: 20° extension to 20° flexion
                2 points: >20° extension or 20-45° flexion
                3 points: 45-90° flexion
                4 points: >90° flexion
                
                Provide a detailed analysis of what you observe and your base score as a numeric value.""",
            "upper_arm_adj1": """Is the shoulder raised in this image? Answer with yes or no and explain your observation.""",
            "upper_arm_adj2": """Is the upper arm abducted in this image? Answer with yes or no and explain your observation.""",
            "upper_arm_adj3": """Is the arm supported or is the person leaning in this image? Answer with yes or no and explain your observation.""",
            "lower_arm": """Analyze the lower arm position and assign a RULA base score based on these criteria:
                1 point: 60-100° flexion
                2 points: <60° or >100° flexion
                
                Provide a detailed analysis of what you observe and your base score as a numeric value.""",
            "lower_arm_adj1": """Is the arm working across midline or out to side in this image? Answer with yes or no and explain your observation.""",
            "wrist": """Analyze the wrist position and assign a RULA base score based on these criteria:
                1 point: Neutral position
                2 points: 0-15° flexion/extension
                3 points: >15° flexion/extension
                
                Provide a detailed analysis of what you observe and your base score as a numeric value.""",
            "wrist_adj1": """Is the wrist bent from midline (deviated) in this image? Answer with yes or no and explain your observation.""",
            "neck": """Analyze the neck position and assign a RULA base score based on these criteria:
                1 point: 0-10° flexion
                2 points: 10-20° flexion
                3 points: >20° flexion
                4 points: In extension
                
                Provide a detailed analysis of what you observe and your base score as a numeric value.""",
            "neck_adj1": """Is the neck twisted in this image? Answer with yes or no and explain your observation.""",
            "neck_adj2": """Is the neck side bending in this image? Answer with yes or no and explain your observation.""",
            "trunk": """Analyze the trunk position and assign a RULA base score based on these criteria:
                1 point: Sitting well supported at 90°
                2 points: 0-20° flexion
                3 points: 20-60° flexion
                4 points: >60° flexion
                
                Provide a detailed analysis of what you observe and your base score as a numeric value.""",
            "trunk_adj1": """Is the trunk twisted in this image? Answer with yes or no and explain your observation.""",
            "trunk_adj2": """Is the trunk side bending in this image? Answer with yes or no and explain your observation.""",
            "legs": """Analyze the legs position and assign a RULA score based on these criteria:
                1 point: Legs and feet well supported and in balanced position
                2 points: Legs or feet not properly supported or in unbalanced position
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
            "wrist_twist": """Analyze the wrist twist in this image and assign a RULA score based on these criteria:
                1 point: Mainly in mid-range of twist
                2 points: At or near end of twisting range
                
                Provide a detailed analysis of what you observe and your score as a numeric value.""",
        }
        return base_prompts.get(component, "")

    def _get_adjustment_value(self, image_path: str, component_adj: str) -> dict:
        """Get adjustment value for a specific component using Gemini Vision."""
        prompt = (
            self._get_component_prompt(component_adj)
            + '\n\nReturn your response as a JSON with this structure: {"analysis": "your detailed analysis", "applies": true/false}'
        )

        try:
            # Load the image
            image = PIL.Image.open(image_path)

            # Generate content with structured output
            response = self.model.generate_content(
                contents=[prompt, image],
                generation_config={
                    "response_mime_type": "application/json",
                },
            )

            # Parse the response
            import json

            result = json.loads(response.text)
            return result
        except Exception as e:
            print(f"Error getting adjustment for {component_adj}: {e}")
            return {
                "analysis": f"Error analyzing {component_adj}",
                "applies": False,
            }

    def _get_component_score(self, image_path: str, component: str) -> dict:
        """Get RULA base score and analysis for a specific component using Gemini Vision."""
        prompt = (
            self._get_component_prompt(component)
            + '\n\nReturn your response as a JSON with this structure: {"analysis": "your detailed analysis", "score": numeric_score}'
        )

        try:
            # Load the image
            image = PIL.Image.open(image_path)

            # Generate content with structured output
            response = self.model.generate_content(
                contents=[prompt, image],
                generation_config={
                    "response_mime_type": "application/json",
                },
            )

            # Parse the response
            import json

            result = json.loads(response.text)
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
        adjustments = {}

        # Define components and their adjustments
        component_adjustments = {
            "upper_arm": ["upper_arm_adj1", "upper_arm_adj2", "upper_arm_adj3"],
            "lower_arm": ["lower_arm_adj1"],
            "wrist": ["wrist_adj1"],
            "neck": ["neck_adj1", "neck_adj2"],
            "trunk": ["trunk_adj1", "trunk_adj2"],
            "legs": [],
            "wrist_twist": [],
        }

        # Process each component and its adjustments
        for component, adj_list in component_adjustments.items():
            # Get base score
            result = self._get_component_score(image_path, component)
            base_score = result["score"]
            analyses[component] = result["analysis"]

            # Process adjustments
            adj_total = 0
            component_adjustments_details = {}

            for adj in adj_list:
                adj_result = self._get_adjustment_value(image_path, adj)
                applies = adj_result["applies"]

                # Special case for upper_arm_adj3 which is a -1 adjustment
                if adj == "upper_arm_adj3" and applies:
                    adj_value = -1
                else:
                    adj_value = 1 if applies else 0

                component_adjustments_details[adj] = {
                    "analysis": adj_result["analysis"],
                    "applies": applies,
                    "value": adj_value,
                }
                adj_total += adj_value

            # Calculate final score
            final_score = max(1, base_score + adj_total)  # Ensure score is at least 1

            # Store results
            scores[f"{component}_score"] = final_score
            adjustments[component] = component_adjustments_details

        return {
            "scores": scores,
            # "analyses": analyses,
            # "adjustments": adjustments,
        }

    def get_model_type(self) -> str:
        """Return model type."""
        return "score-based"
