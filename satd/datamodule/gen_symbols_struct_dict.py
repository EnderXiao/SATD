import os
import glob
from tqdm import tqdm
from functools import lru_cache
from typing import Dict, List

label_path = 'test_hyb'


@lru_cache()
def default_dict():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "word.txt")


class CROHMEVocab:

    PAD_IDX = 0
    EOS_IDX = 1
    SOS_IDX = 2
    STRUCT = 3

    def __init__(self, dict_path: str = default_dict()) -> None:
        self.word2idx = dict()
        # self.word2idx["<pad>"] = self.PAD_IDX
        # self.word2idx["<eos>"] = self.EOS_IDX
        # self.word2idx["<sos>"] = self.SOS_IDX
        # self.word2idx["struct"] = self.STRUCT
        self.labels = glob.glob(os.path.join(label_path, '*.txt'))

        with open(dict_path, 'r') as f:
            for line in f.readlines():
                w = line.strip()
                self.word2idx[w] = len(self.word2idx)
        # with open(dict_path, "w") as writer:
        #     writer.write('<pad>\n<eos>\n<sos>\nstruct\n')
        #     i = 4
        #     for item in tqdm(self.labels):
        #         with open(item) as f:
        #             lines = f.readlines()
        #         for line in lines:
        #             cid, c, pid, p, *r = line.strip().split()
        #             if c not in self.word2idx:
        #                 writer.write(f'{c}\n')
        #                 self.word2idx[c] = i
        #                 i += 1
        #     writer.write('above\nbelow\nsub\nsup\nl-sup\ninside\nright')

        self.idx2word: Dict[int,
                            str] = {v: k
                                    for k, v in self.word2idx.items()}

    def words2indices(self, words: List[str]) -> List[int]:
        return [self.word2idx[int(w)] for w in words]

    def indices2words(self, id_list: List[str]) -> List[str]:
        # print("vocab id2word: ", id_list)
        # print(self.idx2word[24])
        # tensor转换int
        res = []
        # for i in id_list:
            # if int(i) in self.idx2word.keys():
            #     res.append(self.idx2word[int(i)])
            # else:
            #     res.append("\\Pi")
        return [self.idx2word[int(i)] for i in id_list]
        # return res

    def indices2label(self, id_list: List[int]) -> str:
        # print("vocab id_list: ", id_list)
        words = self.indices2words(id_list)
        # print("vocab id2label: ", words)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


vocab = CROHMEVocab()

if __name__ == "__main__":
    labels = glob.glob(os.path.join(label_path, '*.txt'))
    print(labels)

    words_dict = set(['<pad>', '<eos>', '<sos>', 'struct'])

    with open('word.txt', 'w') as writer:
        writer.write('<pad>\n<eos>\n<sos>\nstruct\n')
        i = 4
        for item in tqdm(labels):
            with open(item) as f:
                lines = f.readlines()
            for line in lines:
                cid, c, pid, p, *r = line.strip().split()
                if c not in words_dict:
                    words_dict.add(c)
                    writer.write(f'{c}\n')
                    i += 1
        # writer.write('above\nbelow\nsub\nsup\ns-sub\ns-sup\nl-sup\ninside\nright')
        writer.write('above\nbelow\nsub\nsup\nl-sup\ninside\nright')
    print(i)
