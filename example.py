from satd.datamodule.gen_symbols_struct_dict import vocab
from satd.lit_satd import LitSATD
from torchvision.transforms import ToTensor
import torch
from PIL import Image
# from IPython.display import display'
from satd.utils.generation_utils import prediction
from satd.utils.generation_utils import my_convert

ckpt = './lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate=0.6294.ckpt'

model = LitSATD.load_from_checkpoint(ckpt)
model = model.eval()
device = torch.device("cpu")
model = model.to(device)

img = Image.open('./example/18_em_1.bmp')
# img.show()

img = ToTensor()(img)
mask = torch.zeros_like(img, dtype=torch.bool)
hyp = model.approximate_joint_search(img.unsqueeze(0), mask)
hyp = hyp[0]
pred_latex_child = vocab.indices2label(hyp.seq)
# structHyb = structHyb > 0.5
# structList = structHyb.tolist()
print(pred_latex_child)
result = my_convert(hyp.seq)
print(result)
