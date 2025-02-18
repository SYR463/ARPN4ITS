import re


# 解析树结构并存储上下文信息
def parse_tree(text):
    lines = text.strip().split("\n")
    stack = []
    result = []

    for line in lines:
        # 计算缩进级别
        indent_level = len(line) - len(line.lstrip())

        # 创建节点信息
        if "mbr" in line:
            node = {"type": "mbr", "line": line.strip(), "children": [], "siblings": []}
        elif "entry" in line:
            node = {"type": "entry", "line": line.strip(), "children": [], "siblings": []}

        # 确定节点的父节点
        while stack and stack[-1]["indent_level"] >= indent_level:
            stack.pop()  # 出栈返回父节点

        # 如果栈不空，当前栈顶是父节点
        if stack:
            parent_node = stack[-1]["node"]
            node["parent"] = parent_node
            parent_node["children"].append(node)
            # 为当前父节点的所有兄弟节点设置兄弟关系
            if len(parent_node["children"]) > 1:
                siblings = parent_node["children"][:-1]  # 除去当前节点后的所有兄弟
                for sibling in siblings:
                    sibling["siblings"].append(node)
                    node["siblings"].append(sibling)

        # 将当前节点进栈
        stack.append({"node": node, "indent_level": indent_level})

        # 将当前节点保存
        result.append(node)

    return result


# 映射函数：将x坐标转化为字母
def map_x_to_letter(x):
    # 假设x从0.0开始递增，每增加1.0，就对应下一个字母
    # 例如，0.0 -> 'A', 1.0 -> 'B', 2.0 -> 'C'，依此类推
    return chr(int(x) + 65)  # 'A'的ASCII值是65


# 生成token的函数
def generate_tokens(rect_string):
    # 从字符串中提取x1, y1, x2, y2
    match = re.search(r"x1=([\d.]+), y1=([\d.]+), x2=([\d.]+), y2=([\d.]+)", rect_string)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())

        # 转换坐标为token
        token1 = f"{map_x_to_letter(x1)}{int(y1)}"
        token2 = f"{map_x_to_letter(x2)}{int(y2)}"

        return token1, token2
    return None, None


# 打印节点信息
def print_node_info(node):
    print(f"Node: {node['line']}")
    print(f"Parent: {node['parent']['line'] if 'parent' in node else 'None'}")
    print(f"Siblings: {[sibling['line'] for sibling in node['siblings']]}")
    print(f"Children: {[child['line'] for child in node['children']]}")
    print("-" * 50)


# 将文件转换为token的形式
def convert_file_to_tokens(text):
    lines = text.strip().split("\n")
    tokenized_lines = []

    for line in lines:
        if "mbr" in line or "entry" in line:
            # 直接生成token
            token1, token2 = generate_tokens(line)
            if token1 and token2:
                tokenized_line = f"{token1},{token2}"
                tokenized_lines.append(tokenized_line)
    return tokenized_lines


if __name__ == '__main__':
    # 示例树形结构文本
    text = """
    mbr=Rectangle [x1=0.0, y1=4.0, x2=4.0, y2=26.0]
      mbr=Rectangle [x1=0.0, y1=4.0, x2=4.0, y2=6.0]
        mbr=Rectangle [x1=0.0, y1=7.0, x2=4.0, y2=14.0]
          entry=Entry [value=27, geometry=Rectangle [x1=0.0, y1=4.0, x2=1.0, y2=5.0]]
          entry=Entry [value=48, geometry=Rectangle [x1=3.0, y1=4.0, x2=4.0, y2=5.0]]
          entry=Entry [value=34, geometry=Rectangle [x1=2.0, y1=5.0, x2=3.0, y2=6.0]]
        mbr=Rectangle [x1=0.0, y1=7.0, x2=4.0, y2=15.0]
      mbr=Rectangle [x1=0.0, y1=7.0, x2=4.0, y2=12.0]
        entry=Entry [value=35, geometry=Rectangle [x1=3.0, y1=7.0, x2=4.0, y2=8.0]]
        entry=Entry [value=32, geometry=Rectangle [x1=1.0, y1=8.0, x2=2.0, y2=9.0]]
        entry=Entry [value=25, geometry=Rectangle [x1=2.0, y1=9.0, x2=3.0, y2=10.0]]
        entry=Entry [value=23, geometry=Rectangle [x1=0.0, y1=10.0, x2=1.0, y2=11.0]]
        entry=Entry [value=20, geometry=Rectangle [x1=1.0, y1=10.0, x2=2.0, y2=11.0]]
        entry=Entry [value=31, geometry=Rectangle [x1=3.0, y1=10.0, x2=4.0, y2=11.0]]
        entry=Entry [value=26, geometry=Rectangle [x1=1.0, y1=11.0, x2=2.0, y2=12.0]]
    """

    # 将文件内容转化为token
    tokenized_lines = convert_file_to_tokens(text)

    # 解析树形结构
    parsed_tree = parse_tree(tokenized_lines)

    # 查找并打印某一节点的上下文信息
    for node in parsed_tree:
        # 这里示范打印每个节点的信息，可以指定某个节点来查看
        print_node_info(node)
