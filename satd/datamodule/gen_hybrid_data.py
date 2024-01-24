import os
from tqdm import tqdm


class Tree:

    def __init__(self,
                 label,
                 parent_label='None',
                 id=0,
                 parent_id=0,
                 op='none'):
        self.children = []
        self.label = label
        self.id = id
        self.parent_id = parent_id
        self.parent_label = parent_label
        self.op = op


def convert(root: Tree, f):
    if root.tag == 'N-T':
        f.write(
            f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n'
        )
        for child in root.children:
            convert(child, f)
    else:
        f.write(
            f'{root.id}\t{root.label}\t{root.parent_id}\t{root.parent_label}\t{root.tag}\n'
        )


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


def gen(state):
    label = '../train_latex.txt'
    out = 'train_hyb'

    if state == 'test':
        label = '../test_latex.txt'
        out = 'test_hyb'

    position = set(['^', '_'])
    math = set(['\\frac', '\sqrt'])

    with open(label) as f:
        lines = f.readlines()
    num = 0
    for line in tqdm(lines):
        # line = 'RIT_2014_178.jpg x ^ { \\frac { p } { q } } = \sqrt [ q ] { x ^ { p } } = \sqrt [ q ] { x ^ { p } }'

        name, *words = line.split()
        name = name.split('.')[0]
        parents = []
        root = Tree('root', parent_label='root', parent_id=-1)

        struct_list = ['\\frac', '\\sqrt']

        labels = []
        id = 1
        parents = [Tree('<sos>', id=0)]
        parent = Tree('<sos>', id=0)
        words = formate_latex(words)
        for i in range(len(words)):
            a = words[i]
            # 限制公式上下标是否压缩，\limits表示不压缩，默认采用\limits的形式，因此跳过
            if a == '\\limits':
                continue
            # 开头不能是这些字符
            if i == 0 and words[i] in ['_', '^', '{', '}']:
                print(name)
                continue

            elif words[i] == '{':
                if words[i - 1] == '\\frac':
                    labels.append([id, 'struct', parent.id, parent.label])
                    parents.append(Tree('\\frac', id=parent.id, op='above'))
                    id += 1
                    parent = Tree('above', id=parents[-1].id + 1)
                elif words[i - 1] == '}' and parents[
                    -1].label == '\\frac' and parents[-1].op == 'above':
                    parent = Tree('below', id=parents[-1].id + 1)
                    parents[-1].op = 'below'

                elif words[i - 1] == '\\sqrt':
                    labels.append([id, 'struct', parent.id, '\\sqrt'])
                    parents.append(Tree('\\sqrt', id=parent.id))
                    parent = Tree('inside', id=id)
                    id += 1
                elif words[i - 1] == ']' and parents[-1].label == '\\sqrt':
                    parent = Tree('inside', id=parents[-1].id + 1)

                elif words[i - 1] == '^':
                    if words[i - 2] != '}':
                        if words[i - 2] == '\\sum':
                            labels.append(
                                [id, 'struct', parent.id, parent.label])
                            parents.append(Tree('\\sum', id=parent.id))
                            parent = Tree('above', id=id)
                            id += 1

                        else:
                            labels.append(
                                [id, 'struct', parent.id, parent.label])
                            parents.append(Tree(words[i - 2], id=parent.id))
                            parent = Tree('sup', id=id)
                            id += 1

                    else:
                        # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                        if parents[-1].label == '\\sum':
                            parent = Tree('above', id=parents[-1].id + 1)
                        else:
                            # parent = Tree('s-sup', id=parents[-1].id + 1)
                            parent = Tree('sup', id=parents[-1].id + 1)
                        # parents.append(parent)
                        # id += 1

                elif words[i - 1] == '_':
                    if words[i - 2] != '}':
                        if words[i - 2] == '\\sum':
                            labels.append(
                                [id, 'struct', parent.id, parent.label])
                            parents.append(Tree('\\sum', id=parent.id))
                            parent = Tree('below', id=id)
                            id += 1

                        else:
                            labels.append(
                                [id, 'struct', parent.id, parent.label])
                            parents.append(Tree(words[i - 2], id=parent.id))
                            parent = Tree('sub', id=id)
                            id += 1

                    else:
                        # labels.append([id, 'struct', parents[-1].id, parents[-1].label])
                        if parents[-1].label == '\\sum':
                            parent = Tree('below', id=parents[-1].id + 1)
                        else:
                            # parent = Tree('s-sub', id=parents[-1].id + 1)
                            parent = Tree('sub', id=parents[-1].id + 1)
                        # id += 1
                else:
                    # parent = Tree('right', id=parents[-1].id + 1)
                    # id += 1
                    print('unknown word before {', name, i)

            elif words[i] == '[' and words[i - 1] == '\\sqrt':
                labels.append([id, 'struct', parent.id, '\\sqrt'])
                parents.append(Tree('\\sqrt', id=parent.id))
                parent = Tree('l-sup', id=id)
                id += 1
            elif words[i] == ']' and parents[-1].label == '\\sqrt':
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

            elif words[i] == '}':
                # if words[i - 1] != '}':
                #     labels.append([id, '<eos>', parent.id, parent.label])
                #     id += 1
                labels.append([id, '<eos>', parent.id, parent.label])
                id += 1

                if i + 1 < len(words) and words[i + 1] == '{' and parents[
                    -1].label == '\\frac' and parents[-1].op == 'above':
                    continue
                if i + 1 < len(words) and words[i + 1] in ['_', '^']:
                    continue
                # elif i + 1 < len(words) and words[i + 1] != '}':
                #     parent = Tree('right', id=parents[-1].id + 1)
                elif i + 1 < len(words):
                    parent = Tree('right', id=parents[-1].id + 1)
                parents.pop()

            else:
                if words[i] in ['^', '_']:
                    continue
                labels.append([id, words[i], parent.id, parent.label])
                parent = Tree(words[i], id=id)
                id += 1

        parent_dict = {0: []}
        for i in range(len(labels)):
            parent_dict[i + 1] = []
            parent_dict[labels[i][2]].append(labels[i][3])

        with open(out + f'/{name}.txt', 'w') as f:
            for line in labels:
                id, label, parent_id, parent_label = line
                if label != 'struct':
                    f.write(
                        f'{id}\t{label}\t{parent_id}\t{parent_label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n'
                    )
                else:
                    tem = f'{id}\t{label}\t{parent_id}\t{parent_label}'
                    tem = tem + '\tabove' if 'above' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tbelow' if 'below' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tsub' if 'sub' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tsup' if 'sup' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tl-sup' if 'l-sup' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tinside' if 'inside' in parent_dict[
                        id] else tem + '\tNone'
                    tem = tem + '\tright' if 'right' in parent_dict[
                        id] else tem + '\tNone'
                    f.write(tem + '\n')
            if label != '<eos>':
                f.write(
                    f'{id + 1}\t<eos>\t{id}\t{label}\tNone\tNone\tNone\tNone\tNone\tNone\tNone\n'
                )


if __name__ == '__main__':
    gen('train')
    gen('test')
