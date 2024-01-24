import torch

from satd.lit_satd import LitSATD

from pytorch_lightning import Trainer
from satd.datamodule.datamodule import CROHMEDatamodule

test_year = '2014'
config_path = "config.yaml"

# 加载模型的权重
checkpoint = torch.load('lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate=0.6294.ckpt')

# 获取模型的 state_dict
state_dict = checkpoint['state_dict']

# 遍历 state_dict 并替换层级名称
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('comer_model'):
        new_key = key.replace('comer_model', 'satd_model')
        new_key = new_key.replace('arm', 'maam')
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
checkpoint['state_dict'] = new_state_dict
torch.save(checkpoint, 'lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate2=0.6294.ckpt')

# print(new_state_dict)
# # 更新模型的 state_dict
# trainer = Trainer(logger=False, gpus=1)
# dm = CROHMEDatamodule(config_path=config_path, test_year=test_year)
# model = LitSATD.load_state_dict(state_dict=new_state_dict)
# trainer.fit(model, datamodule=dm)
# trainer.save_checkpoint()



# # # 保存修改后的模型
# # torch.save(model.state_dict(), 'path/to/your/modified_model.ckpt')
# torch.save(model.state_dict(), 'lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate2=0.6294.ckpt')