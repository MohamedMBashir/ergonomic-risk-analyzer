# from models.gpt_vision.gpt_vision_model import GPTVisionModel
from models.gpt_vision.gpt_vision_adj_model import GPTVisionAdjModel
from models.gemini_vision.gemini_vision_adj_model import GeminiVisionAdjModel
from models.mppe.mppe_model import Pose3DMPPEModel
from models.pose3d.pose3d_model import Pose3DModel

gpt_vision_adj_model = GPTVisionAdjModel()
gemini_vision_adj_model = GeminiVisionAdjModel()
mppe_model = Pose3DMPPEModel()
pose3d_model = Pose3DModel()

output_gpt = gpt_vision_adj_model.process_image("./inputs/input_old_man.jpeg")
output_gemini = gemini_vision_adj_model.process_image("./inputs/input_old_man.jpeg")
output_mppe = mppe_model.process_image("./inputs/input_old_man.jpeg")
output_pose3d = pose3d_model.process_image("./inputs/input_old_man.jpeg")

print("\n\nGPTVisionAdjModel Output:\n", output_gpt)
print("\n\nGeminiVisionAdjModel Output:\n", output_gemini)
print("\n\nPose3DMPPEModel Output:\n", output_mppe)
print("\n\nPose3DModel Output:\n", output_pose3d)
