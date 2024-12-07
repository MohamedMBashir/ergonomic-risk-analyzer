from typing import Dict

class RULAAssessment:
    def assess(self, angles: Dict[str, float]) -> Dict:
        """Perform RULA assessment based on calculated angles"""
        scores = {
            'upper_arm_score': self._score_upper_arm(angles['upper_arm_right']),
            'lower_arm_score': self._score_lower_arm(angles['lower_arm_right']),
            'neck_score': self._score_neck(angles['neck']),
            'trunk_score': self._score_trunk(angles['trunk']),
            # Add more scores
        }
        
        final_score = self._calculate_final_score(scores)
        
        return {
            'detailed_scores': scores,
            'final_score': final_score,
            'risk_level': self._get_risk_level(final_score)
        }
    
    def _score_upper_arm(self, angle: float) -> int:
        """Score upper arm position"""
        if angle <= 20:
            return 1
        elif angle <= 45:
            return 2
        elif angle <= 90:
            return 3
        else:
            return 4
    
    # Add more scoring methods