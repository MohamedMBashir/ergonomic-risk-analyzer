import numpy as np

def calc_upper_arm_score(angle, adjustments={}): # ✅ Revised
    if 20 >= angle >= -20:
        score = 1
    elif -20 > angle:
        score =  2
    elif 45 >= angle > 20:
        score = 2
    elif 90 >= angle > 45:
        score = 3
    elif 90 < angle:
        score = 4

    if adjustments.get("shoulder_raised"): # TODO: Implement this in AngleCalculator
        score += 1

    if adjustments.get("upper_arm_adducted"): # TODO: Implement this in AngleCalculator
        score += 1

    if adjustments.get("arm_supported_or_leaning"): # TODO: Implement this in AngleCalculator
        score -= 1

    return score


def calc_lower_arm_score(angle, adjustments={}): # ✅ Revised
    if 100 >= angle >= 60:
        score = 1
    elif angle > 100 or angle < 60:
        score = 2

    if adjustments.get("arms_outside_inside"): # TODO: Implement this in AngleCalculator
        score += 1
    
    return score
    

def calc_wrist_score(angle, adjustments={}): # ✅ Revised
    if 1>= angle >= -1:
        score = 1
    elif 15 >= angle >= -15:
        score = 2
    elif angle > 15 or angle < -15:
        score = 3
    
    if adjustments.get("wrist_bend"): # TODO: Implement this in AngleCalculator
        if adjustments.get("wrist_bend") > 10:
            score += 1

    return score

def calc_wrist_twist_score(angle, adjustments={}): # ❌ Not Straightforward. TODO: Check with AngleCalculator
    if -45 <= angle <= 45:
        return 1
    else:
        return 2

def calc_neck_score(angle, adjustments={}): # ✅ Revised
    if 0 <= angle <= 10:
        score = 1
    elif 10 < angle <= 20:
        score = 2
    elif 20 < angle:
        score = 3
    elif angle < 0:
        score = 4

    if adjustments.get("neck_twist"): # TODO: Implement this in AngleCalculator
        score += 1
    
    if adjustments.get("neck_bended"): # TODO: Implement this in AngleCalculator
        score += 1

    return score

def calc_trunk_score(angle, adjustments={}): # ✅ Revised
    if 0 <= angle <= 2:
        score = 1
    elif 2 < angle <= 20:
        score = 2
    elif 20 < angle <= 60:
        score = 3
    elif 60 < angle:
        score = 4
    
    if adjustments.get("trunk_twisted"): # TODO: Implement this in AngleCalculator
        score += 1

    if adjustments.get("trunk_bended"): # TODO: Implement this in AngleCalculator
        score += 1

    return score

def calc_leg_score(angle, adjustments={}): # ✅ Revised
    if angle < 10:  
        return 1
    else:
        return 2

def get_table_a_score(upper_arm, lower_arm, wrist, wrist_twist): # ✅ Revised
    table_a = np.array([
        [[[1,2],[2,2],[2,3],[3,3]],
         [[2,2],[2,2],[3,3],[3,3]],
         [[2,3],[3,3],[3,3],[4,4]]],

        [[[2,3],[3,3],[3,4],[4,4]],
         [[3,3],[3,3],[3,4],[4,4]],
         [[3,4],[4,4],[4,4],[5,5]]],

        [[[3,3],[4,4],[4,4],[5,5]],
         [[3,4],[4,4],[4,4],[5,5]],
         [[4,4],[4,4],[4,5],[5,5]]],

        [[[4,4],[4,4],[4,5],[5,5]],
         [[4,4],[4,4],[4,5],[5,5]],
         [[4,4],[4,5],[5,5],[6,6]]],

        [[[5,5],[5,5],[5,6],[6,7]],
         [[5,6],[6,6],[6,7],[7,7]],
         [[6,6],[6,7],[7,7],[7,8]]],

        [[[7,7],[7,7],[7,8],[8,9]],
         [[8,8],[8,8],[8,9],[9,9]],
         [[9,9],[9,9],[9,9],[9,9]]]
    ])
    return table_a[upper_arm-1][lower_arm-1][wrist-1][wrist_twist-1]

def get_table_b_score(neck, trunk, legs): # ✅ Revised
    table_b = np.array([
        [[1,3],[2,3],[3,4],[5,5],[6,6],[7,7]],
        [[2,3],[2,3],[4,5],[5,5],[6,7],[7,7]],
        [[3,3],[3,4],[4,5],[5,6],[6,7],[7,7]],
        [[5,5],[5,6],[6,7],[7,7],[7,7],[8,8]],
        [[7,7],[7,7],[7,8],[8,8],[8,8],[8,8]],
        [[8,8],[8,8],[8,8],[8,9],[9,9],[9,9]]
    ])
    return table_b[neck-1][trunk-1][legs-1]

def get_table_c_score(score_a, score_b): # ✅ Revised
    
    if score_a > 8:
        score_a = 8
    if score_b > 7:
        score_b = 7

    table_c = np.array([
        [1,2,3,3,4,5,5],
        [2,2,3,4,4,5,5],
        [3,3,3,4,4,5,6],
        [3,3,3,4,5,6,6],
        [4,4,4,5,6,7,7],
        [4,4,5,6,6,7,7],
        [5,5,6,6,7,7,7],
        [5,5,6,7,7,7,7]
    ])
    return table_c[score_a-1][score_b-1]

def calculate_rula_score(angles, force_l=1, muscle_u=1): # ✅ Revised
    """
    Calculate RULA score based on joint angles.
    
    :param angles: Dictionary containing joint angles in degrees
    :return: RULA score and intermediate scores
    """
    upper_arm_score = calc_upper_arm_score(angles['upper_arm'])
    lower_arm_score = calc_lower_arm_score(angles['lower_arm'])
    wrist_score = calc_wrist_score(angles['wrist'])
    wrist_twist_score = calc_wrist_twist_score(angles['wrist_twist'])
    
    neck_score = calc_neck_score(angles['neck'])
    trunk_score = calc_trunk_score(angles['trunk'])
    leg_score = calc_leg_score(angles['leg'])
    
    # Assuming minimal force/load and static muscle use
    force_load_score = force_l # NOTE: We need to figure out the proper default
    muscle_use_score = muscle_u # NOTE: We need to figure out the proper default
    
    table_a_score = get_table_a_score(upper_arm_score, lower_arm_score, wrist_score, wrist_twist_score)
    table_b_score = get_table_b_score(neck_score, trunk_score, leg_score)
    
    score_a = table_a_score + force_load_score + muscle_use_score
    score_b = table_b_score + force_load_score + muscle_use_score
    
    final_score = get_table_c_score(score_a, score_b)
    
    return {
        'final_score': final_score,
        'upper_arm_score': upper_arm_score,
        'lower_arm_score': lower_arm_score,
        'wrist_score': wrist_score,
        'wrist_twist_score': wrist_twist_score,
        'neck_score': neck_score,
        'trunk_score': trunk_score,
        'leg_score': leg_score,
        'table_a_score': table_a_score,
        'table_b_score': table_b_score,
        'score_a': score_a,
        'score_b': score_b
    }
    
# Example usage:
angles = {
    'upper_arm': 45,
    'lower_arm': 80,
    'wrist': 10,
    'wrist_twist': 30,
    'neck': 20,
    'trunk': 30,
    'leg': 0
}

# rula_scores = calculate_rula_score(angles)
# print(f"RULA Score: {rula_scores['final_score']}")
# print(f"Detailed Scores: {rula_scores}")