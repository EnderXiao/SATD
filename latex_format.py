

def formate_latex(latex_str):
    block_stack = []
    str = []
    struct_list = ["\\frac", "^", "_", "\\sqrt", "\\sqrt l-sup", "\\frac above", "\\frac below"]
    block_list = ["{", "}", "[", "]"]
    i = 0
    while i < len(latex_str):
        word_now = latex_str[i]
        if word_now in struct_list:
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
            if word_now == "[":
                str += ["["]
                if len(block_stack):
                    if block_stack[-1] == "\\sqrt":
                        block_stack.append("\\sqrt l-sup")
                        block_stack.append("[")
            if word_now == "}":
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
            if word_now == "]":
                if len(block_stack):
                    str += ["]"]
                    if block_stack[-1] == "[":
                        block_stack.pop()
                        block_stack.pop()
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


if __name__ == "__main__":
    x = input()
    x = x.split(" ")
    res = formate_latex(x)
    print(res)
