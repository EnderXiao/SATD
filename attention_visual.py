import matplotlib.pyplot as plt

from satd.datamodule.gen_symbols_struct_dict import vocab
from satd.lit_satd import LitSATD
from torchvision.transforms import ToTensor
import torch
import skimage.transform
import matplotlib.cm as cm
from PIL import Image
import numpy as np
# from IPython.display import display'
from satd.utils.generation_utils import my_convert


ckp_path = "C:\\Net\\SATD\\lightning_logs\\version_0\\checkpoints\\epoch=203-step=225215-val_ExpRate=0.6294.ckpt"

model = LitSATD.load_from_checkpoint(ckp_path)
model = model.eval()
device = torch.device("cpu")
model = model.to(device)

imgBase = Image.open('./example/200922-1017-12.bmp')  # 树形结构展示
# imgBase = Image.open('./example/200922-1017-74.bmp')  # 多重复结构
# imgBase = Image.open('./example/26_em_99.bmp')  # 多重复字符串
# imgBase = Image.open('./example/70_hirata.bmp')  # 多重复字符串
# img.show()
scale_size = 2
img = ToTensor()(imgBase)
mask = torch.zeros_like(img, dtype=torch.bool)
hyp, tree_attn, word_attn = model.approximate_joint_search(img.unsqueeze(0), mask)
l2r_tree_attn = tree_attn[0]
print(l2r_tree_attn.shape)
l2r_word_attn = word_attn[0]
num_head = l2r_word_attn.shape[0]
image = imgBase.resize([14 * 24, 14 * 24], Image.LANCZOS)
hyp = hyp[0]
pred_latex_child = vocab.indices2label(hyp.seq)
pred_child_list = pred_latex_child.split(" ")
print("seq len: ", len(pred_child_list))
fig = plt.figure(figsize=(5, 5))
for t in range(len(pred_child_list)):
    if t > 100:
        break
    print(pred_child_list[t])
    print(l2r_tree_attn.shape)
    print(l2r_word_attn.shape)
    current_l2r_word_alpha = l2r_word_attn[3, t]
    print(current_l2r_word_alpha.shape)
    word_alpha = skimage.transform.resize(current_l2r_word_alpha.detach().numpy(), [14 * 24, 14 * 24])


    plt.subplot(2, int(np.ceil(len(pred_child_list) / 2)), t + 1)
    plt.text(0, 1, '%s' % (pred_child_list[t]), color='black', backgroundcolor='white', fontsize=12)
    plt.imshow(image)
    plt.imshow(word_alpha, alpha=0.8)
    plt.set_cmap(cm.Greys_r)
    plt.axis('off')
    plt.subplots_adjust(hspace=0.1, left=0.1)

result = my_convert(hyp.seq)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.7, wspace=0.05, hspace=0.05)
plt.show()
print(result)
