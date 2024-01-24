import torch
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import cv2
from typing import List, Optional, Tuple
from torch import FloatTensor, LongTensor
from dataclasses import dataclass
from satd.datamodule.transform import ScaleAugmentation, ScaleToLimitRange
import torchvision.transforms as tf

MAX_SIZE = 32e4  # change here accroading to your GPU memory


K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024



@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, 1, H, W]
    indices: LongTensor  # [b, l, d]
    indices_mask: LongTensor  # [b, l, 2]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(img_bases=self.img_bases,
                     imgs=self.imgs.to(device),
                     mask=self.mask.to(device),
                     indices=self.indices,
                     indices_mask=self.indices_mask)


class HYBTr_Dataset(Dataset):

    def __init__(self, params, image_path, label_path, words, is_train=True, scale_aug=True):
        super(HYBTr_Dataset, self).__init__()

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tf.ToTensor(),
        ]
        self.transform = tf.Compose(trans_list)

        with open(image_path, 'rb') as f:
            print("dataSet: ", image_path)
            self.images = pkl.load(f)
        with open(label_path, 'rb') as f:
            self.labels = pkl.load(f)

        self.name_list = list(self.labels.keys())
        self.words = words
        self.max_width = params['image_width']
        self.is_train = is_train
        self.params = params
        self.image_height = params['image_height']
        self.image_width = params['image_width']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        name = self.name_list[idx]

        image = self.images[name]

        # image = torch.Tensor(image) / 255
        image = self.transform(image)
        # image = image.unsqueeze(0)

        label = self.labels[name]

        child_words = [item.split()[1] for item in label]
        child_words = self.words.encode(child_words)
        child_words = torch.LongTensor(child_words)
        child_ids = [int(item.split()[0]) for item in label]
        child_ids = torch.LongTensor(child_ids)

        parent_words = [item.split()[3] for item in label]
        parent_words = self.words.encode(parent_words)
        parent_words = torch.LongTensor(parent_words)
        parent_ids = [int(item.split()[2]) for item in label]
        parent_ids = torch.LongTensor(parent_ids)
        struct_label = [item.split()[4:] for item in label]
        if len(struct_label) == 0:
            print(name)
        struct = torch.zeros((len(struct_label), len(struct_label[0]))).long()
        for i in range(len(struct_label)):
            for j in range(len(struct_label[0])):
                struct[i][j] = struct_label[i][j] != 'None'

        label = torch.cat([
            child_ids.unsqueeze(1),
            child_words.unsqueeze(1),
            parent_ids.unsqueeze(1),
            parent_words.unsqueeze(1), struct
        ],
            dim=1)

        return name, image, label

    def collate_fn(self, batch_images):
        max_width, max_height, max_length = 0, 0, 0
        batch, channel = len(batch_images), batch_images[0][1].shape[0]
        names = []
        proper_items = []
        for item in batch_images:
            fname = item[0]
            names.append(fname)
            item = item[1:]
            if item[0].shape[
                1] * max_width > self.image_width * self.image_height or item[
                0].shape[
                2] * max_height > self.image_width * self.image_height:
                # print(f"image: {fname} size: {item[0].shape[1]} x {item[0].shape[2]} =  bigger than {self.image_width * self.image_height}, ignore")
                continue
            size = item[0].shape[1] * item[0].shape[2]
            if size > MAX_SIZE:
                print(
                    f"image: {fname} size: {item[0].shape[1]} x {item[0].shape[2]} =  bigger than {MAX_SIZE}, ignore"
                )
                continue
            max_height = item[0].shape[
                1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[
                2] if item[0].shape[2] > max_width else max_width
            max_length = item[1].shape[
                             0] - 1 if item[1].shape[0] > max_length else max_length
            proper_items.append(item)

        images, image_masks = torch.zeros(
            (len(proper_items), channel, max_height, max_width)), torch.ones(
            (len(proper_items), max_height, max_width), dtype=torch.bool)
        labels, labels_masks = torch.zeros(
            (len(proper_items), max_length, 11)).long(), torch.ones(
            (len(proper_items), max_length, 2), dtype=torch.bool)

        for i in range(len(proper_items)):

            _, h, w = proper_items[i][0].shape
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:h, :w] = 0

            l = proper_items[i][1].shape[0]
            for j in range(l - 1):
                labels[i][j][0] = proper_items[i][1][j][0]
                labels[i][j][1] = proper_items[i][1][j][1]
                labels[i][j][2] = proper_items[i][1][j + 1][2]
                labels[i][j][3] = proper_items[i][1][j + 1][3]
                labels[i][j][4:] = proper_items[i][1][j][4:]
            # labels[i][:l, :] = proper_items[i][1]
            labels_masks[i][:l, 0] = 1

            for j in range(l - 1):
                labels_masks[i][j][1] = proper_items[i][1][j][4:].sum() != 0

        return Batch(names, images, image_masks, labels, labels_masks)


def get_test_dataset(params):
    words = Words(params['word_path'])
    # print("get_dataset: word_len: ", len(words))

    params['word_num'] = len(words)
    params['struct_num'] = 7
    print(
        f"training data，images: {params['test_image_path']} labels: {params['test_label_path']}"
    )
    test_dataset = HYBTr_Dataset(params, params['test_image_path'], params['test_label_path'], words, is_train=False)

    test_sampler = RandomSampler(test_dataset)

    test_loader = DataLoader(test_dataset,
                             batch_size=params['test_batch_size'],
                             sampler=test_sampler,
                             num_workers=params['workers'],
                             collate_fn=test_dataset.collate_fn,
                             pin_memory=True)

    print(
        f'train dataset: {len(test_dataset)} train steps: {len(test_loader)}')

    return test_dataset, test_loader


def get_dataset(params):
    words = Words(params['word_path'])
    # print("get_dataset: word_len: ", len(words))

    params['word_num'] = len(words)
    params['struct_num'] = 7
    print(
        f"training data，images: {params['train_image_path']} labels: {params['train_label_path']}"
    )
    print(
        f"test data，images: {params['eval_image_path']} labels: {params['eval_label_path']}"
    )
    train_dataset = HYBTr_Dataset(params, params['train_image_path'],
                                  params['train_label_path'], words, is_train=True, scale_aug=True)
    eval_dataset = HYBTr_Dataset(params, params['eval_image_path'],
                                 params['eval_label_path'], words, is_train=False, scale_aug=True)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)
    # eval_sampler = RandomSampler(eval_dataset, num_samples=len(eval_dataset))

    train_loader = DataLoader(train_dataset,
                              batch_size=params['train_batch_size'],
                              sampler=train_sampler,
                              num_workers=params['workers'],
                              collate_fn=train_dataset.collate_fn,
                              pin_memory=True)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=1,
                             sampler=eval_sampler,
                             num_workers=params['workers'],
                             collate_fn=eval_dataset.collate_fn,
                             pin_memory=True)

    print(
        f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
        f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)}')

    return train_dataset, eval_dataset, train_loader, eval_loader


class Words:

    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'{len(words)} symbols in total')

        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {
            i: words[i].strip()
            for i in range(len(words))
        }

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        # label_index = []
        # for item in labels:
        #     if item in self.words_dict.keys():
        #         label_index.append(self.words_dict[item])
        #     else:
        #         label_index.append(118)
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = []
        # for item in label_index:
        #     if item in self.words_index_dict.keys():
        #         label.append(self.words_index_dict[int(item)])
        #     else:
        #         label.append("\\Pi")
        label = ' '.join(
            [self.words_index_dict[int(item)] for item in label_index])
        return label
