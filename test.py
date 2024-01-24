from pytorch_lightning import Trainer

from satd.datamodule.datamodule import CROHMEDatamodule

from satd.lit_satd import LitSATD

import numpy

test_year = "2014"
ckp_path = "lightning_logs/version_0/checkpoints/epoch=203-step=225215-val_ExpRate=0.6294.ckpt"
config_path = "config.yaml"



def formate_latex(latex_str):
    block_stack = []
    str = []
    struct_list = ["\\frac", "^", "_", "\\sqrt", "\\sqrt l-sup", "\\frac above", "\\frac below"]
    block_list = ["{", "}", "[", "]"]
    i = 0
    while i < len(latex_str):
        word_now = latex_str[i]
        if word_now == "\\limits":
            i += 1
            continue
        elif word_now in struct_list:
            block_stack.append(word_now)
            if word_now == "\\frac":
                block_stack.append("\\frac above")
            str += [word_now]
        elif word_now in block_list:
            if word_now == "{":
                if len(block_stack):
                    if block_stack[-1] in ["{", "["]:
                        block_stack.append("res {")
                    else:
                        block_stack.append("{")
                        str += ["{"]
                else:
                    block_stack.append("res {")
            elif word_now == "[":
                str += ["["]
                if len(block_stack):
                    if block_stack[-1] == "\\sqrt":
                        block_stack.append("\\sqrt l-sup")
                        block_stack.append("[")
            elif word_now == "}":
                if len(block_stack):
                    if block_stack[-1] == "res {":
                        block_stack.pop()
                    elif block_stack[-1] == "{":
                        block_stack.pop()
                        tmp_struct = block_stack.pop()
                        if tmp_struct == "\\frac above":
                            block_stack.append("\\frac below")
                        if tmp_struct == "\\frac below":
                            block_stack.pop()
                        str += ["}"]
            elif word_now == "]":
                if len(block_stack):
                    str += ["]"]
                    if block_stack[-1] == "[":
                        block_stack.pop()
                        block_stack.pop()
                else:
                    str += ["]"]
        else:
            if len(block_stack):
                if block_stack[-1] in struct_list:
                    str += ["{", word_now, "}"]
                    tmp_struct = block_stack.pop()
                    if tmp_struct == "\\frac above":
                        block_stack.append("\\frac below")
                    elif tmp_struct == "\\frac below":
                        block_stack.pop()
                else:
                    str += [word_now]
            else:
                str += [word_now]
        i += 1
    return str


def cmp_result(label, rec):
    dist_mat = numpy.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = range(len(rec) + 1)
    dist_mat[:, 0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i - 1, j - 1] + (label[i - 1] != rec[j - 1])
            ins_score = dist_mat[i, j - 1] + 1
            del_score = dist_mat[i - 1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)


def process(recfile, labelfile, resultfile):
    total_dist = 0
    total_err1 = 0
    total_err2 = 0
    total_err3 = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
    for key_rec in rec_mat:
        if "100k" in test_year:
            label = label_mat[key_rec + ".jpg"]
        else:
            label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        label = formate_latex(label)
        rec = formate_latex(rec)
        dist, llen = cmp_result(label, rec)
        if dist <= 1:
            total_err1 += 1
        if dist <= 2:
            total_err2 += 1
        if dist <= 3:
            total_err3 += 1
        if dist > 3:
            print("word wrong name: ", key_rec)
            print("word wrong pred: ", " ".join(rec))
            print("word wrong truth: ", " ".join(label))
            # x = input()
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    err1 = float(total_err1) / total_line
    err2 = float(total_err2) / total_line
    err3 = float(total_err3) / total_line
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line

    f_result = open(resultfile, 'w')
    f_result.write('<=err1 {}\n'.format(err1))
    f_result.write('<=err2 {}\n'.format(err2))
    f_result.write('<=err3 {}\n'.format(err3))
    f_result.write('WER {}\n'.format(wer))
    f_result.write('ExpRate {}\n'.format(sacc))
    f_result.close()


if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(config_path=config_path, test_year=test_year)

    model = LitSATD.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)

    process("result_convert.txt", "caption.txt", "ours_" + test_year + "_exp.txt")
