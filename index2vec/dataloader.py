import os
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
        if "<NL>" in line:
            node = {"type": "mbr", "line": line.strip(), "children": [], "siblings": []}
        elif "<L>" in line:
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


# 打印节点信息
def print_node_info(node):
    print(f"Node: {node['line']}")
    print(f"Parent: {node['parent']['line'] if 'parent' in node else 'None'}")
    print(f"Siblings: {[sibling['line'] for sibling in node['siblings']]}")
    print(f"Children: {[child['line'] for child in node['children']]}")
    print("-" * 50)


# 构建节点的上下文信息
def print_node_context_info(node):
    # 提取父节点、兄弟节点和子节点的值
    parent_line = node['parent']['line'] if 'parent' in node else None
    siblings_lines = [sibling['line'] for sibling in node['siblings']] if node['siblings'] else []
    children_lines = [child['line'] for child in node['children']] if node['children'] else []

    # 兄弟节点和子节点，去掉空节点
    siblings_str = " ".join(siblings_lines) if siblings_lines else ""
    children_str = " ".join(children_lines) if children_lines else ""

    # 打印节点信息，输出最终格式
    # print(f"Node: {node['line']}")
    # print(f"Parent: {parent_line if parent_line else 'None'}")
    # print(f"Siblings: {siblings_str}")
    # print(f"Children: {children_str}")
    # print("-" * 50)

    # 一行输出
    # print(f"{node['line']} {parent_line if parent_line else 'None'} {siblings_str} {children_str}")

    result = f"{node['line']} {parent_line if parent_line else 'None'} {siblings_str} {children_str}"
    return result


# 处理文件夹中的每个文件
def process_files(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理文本文件
        if filename.endswith(".txt"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)

            # 读取文件内容
            with open(input_file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # 解析树结构
            nodes = parse_tree(text)

            # 打开输出文件
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                for node in nodes:
                    # 获取节点上下文信息
                    context_info = print_node_context_info(node)
                    # 写入文件
                    f_out.write(context_info + "\n")

            print(f"Processed: {filename} -> {output_file_path}")


# 测试用例
if __name__ == '__main1__':
    # 示例树形结构文本
    text = """
    <NL> A1 E14
      <NL> A1 E6
        <L> A1 B2
        <L> D3 E4
        <L> B4 C5
        <L> A5 B6
        <L> B5 C6
      <NL> B8 E14
        <L> D8 E9
        <L> B9 C10
        <L> C9 D10
        <L> B11 C12
        <L> C13 D14
    """

    # 解析树形结构
    parsed_tree = parse_tree(text)

    # 查找并打印某一节点的上下文信息
    for node in parsed_tree:
        # 这里示范打印每个节点的信息，可以指定某个节点来查看
        print_node_context_info(node)


# 测试用例，从文件读取
if __name__ == '__main__':
    # 输入和输出文件夹路径
    input_folder = "/mnt/d/project/python/ARPN4ITS/dataPreprocess/outputRTreeToken"
    output_folder = "/mnt/d/project/python/ARPN4ITS/dataPreprocess/outputRTreeTokenContext"

    # 执行文件处理
    process_files(input_folder, output_folder)