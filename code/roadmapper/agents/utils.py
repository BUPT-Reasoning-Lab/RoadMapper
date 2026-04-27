import re


def ExtractMarkdownUtil(text: str, code_type: str) -> tuple:
    """Extract markdown code blocks from model output"""
    pattern = rf"```{code_type}\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)
    return tuple(matches)


def SaveFileUtil(file_name: str, file_content: str):
    """Save content to file with UTF-8 encoding"""
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(file_content)

def FormatRectify(roadmap: str) -> tuple:
    """Check roadmap format and remove lines that don't meet requirements"""
    lines = roadmap.split("\n")
    rectified_lines = []
    pattern = re.compile(r"^#+\s+\d+(?:\.\d+)*\s+\[[^\]]+\](\s+<key_step>)?$")
    is_any_wrong_line = False

    for line in lines:
        if not line.strip():
            rectified_lines.append(line)
            continue

        match = pattern.match(line)
        if match:
            rectified_lines.append(line)
        else:
            print(f"[Format error] The line '{line}' is not a valid line")
            is_any_wrong_line = True

    return "\n".join(rectified_lines), is_any_wrong_line

def CheckIndexValidity(roadmap: str) -> bool:
    # Step 1: Split roadmap by lines
    lines = roadmap.split("\n")
    lines = [line for line in lines if line.strip()]
    
    # Step 2: Extract #, index, and title from each line using regex
    pattern = r'^(#{1,6})\s+(\d+(?:\.\d+)*)\s+\[(.+?)\]'
    nodes = []
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            level_marks = match.group(1)  # Extract # marks
            index = match.group(2)        # Extract index (e.g., 2.1.1)
            title = match.group(3)        # Extract title content
            nodes.append({
                "level_marks": level_marks,
                "index": index,
                "title": title,
            })
        else:
            print(f"[Format error]{line} cannot be matched by the pattern")
            return False
    
    # Step 3: Check index validity, i.e., whether the level of # matches the level of index
    for node in nodes:
        node_level = len(node["level_marks"])
        node_index_level = len(node["index"].split("."))
        if node_level != node_index_level:
            print(f"[Format error]{node} level and index level are not consistent")
            return False
        
    # Step 4: Check index continuity, i.e., whether the index levels are continuous
    for i in range(len(nodes)):
        if len(nodes[i]["level_marks"]) == 1:
            pass
        else:
            current_node = nodes[i]
            node_before = nodes[i-1]
            if len(current_node["level_marks"]) == len(node_before["level_marks"]):
                current_node_sibling_index = current_node["index"].split(".")[-1]
                node_before_sibling_index = node_before["index"].split(".")[-1]
                if int(current_node_sibling_index) != int(node_before_sibling_index) + 1:
                    print(f"[Format error]{current_node['index']} is not the next sibling of {node_before['index']}")
                    return False
            elif len(current_node["level_marks"]) == len(node_before["level_marks"]) + 1:
                current_node_parent_index = current_node["index"][:-2]
                if current_node_parent_index != node_before["index"]:
                    print(f"[Format error]{current_node['index']} is not the child of {node_before['index']}")
                    return False
            else:
                # Find the previous node at the same level and check if the order is continuous
                for j in range(i-1, -1, -1):
                    if len(nodes[j]["level_marks"]) == len(current_node["level_marks"]):
                        if int(nodes[j]["index"].split(".")[-1]) != int(current_node["index"].split(".")[-1]) - 1:
                            print(f"[Format error]{nodes[j]['index']} is not the previous sibling of {current_node['index']}")
                            return False
                        break
    
    return True

def calculate_degree_and_depth(roadmap: str) -> tuple:
    """Calculate the average out degree of the roadmap"""
    # Step 1: Split roadmap by lines
    lines = roadmap.split("\n")
    lines = [line for line in lines if line.strip()]
    
    # Step 2: Extract #, index, and title from each line using regex
    pattern = r'^(#{1,6})\s+(\d+(?:\.\d+)*)\s+\[(.+?)\]'
    nodes = []
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            level_marks = match.group(1)  # Extract # marks
            index = match.group(2)        # Extract index (e.g., 2.1.1)
            title = match.group(3)        # Extract title content
            node = {
                "level_marks": level_marks,
                "index": index,
                "title": title,
                "children": [],
            }
            nodes.append(node)
            
            if len(nodes) > 1:
                parent_node_index = node["index"][:-2]
                for each_node in nodes:
                   if each_node["index"] == parent_node_index:
                       each_node["children"].append(node["index"])
        else:
            print(f"[Format error]{line} cannot be matched by the pattern")
    
    # Step 3: Calculate average out degree
    non_leaf_nodes_children_count = []
    for node in nodes:
        if len(node["children"]) > 0:
            non_leaf_nodes_children_count.append(len(node["children"]))
            
    # Step 4: Calculate average depth
    leaf_nodes_depth = []
    for node in nodes:
        if len(node["children"]) == 0:
            leaf_nodes_depth.append(len(node["level_marks"]))
            
    return sum(non_leaf_nodes_children_count) / len(non_leaf_nodes_children_count), sum(leaf_nodes_depth) / len(leaf_nodes_depth)
            
            
def CountInvalidNodes(roadmap: str) -> dict:
    """Count the number and ratio of nodes that do not meet requirements
    
    Args:
        roadmap: Roadmap string
        
    Returns:
        dict: Dictionary containing the following keys
            - total_nodes: Total number of nodes (non-empty lines)
            - invalid_nodes: Number of invalid nodes
            - valid_nodes: Number of valid nodes
            - invalid_ratio: Ratio of invalid nodes
            - invalid_lines: List of invalid lines
    """
    lines = roadmap.split("\n")
    pattern = re.compile(r"^#+\s+\d+(?:\.\d+)*\s+\[[^\]]+\](\s+<key_step>)?$")
    
    total_nodes = 0
    invalid_nodes = 0
    invalid_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        total_nodes += 1
        match = pattern.match(line)
        
        if not match:
            invalid_nodes += 1
            invalid_lines.append(line)
    
    valid_nodes = total_nodes - invalid_nodes
    invalid_ratio = invalid_nodes / total_nodes if total_nodes > 0 else 0.0
    
    return {
        'total_nodes': total_nodes,
        'invalid_nodes': invalid_nodes,
        'valid_nodes': valid_nodes,
        'invalid_ratio': invalid_ratio,
        'invalid_lines': invalid_lines
    }

if __name__ == "__main__":
    roadmap = """
# 1 [Main Step]
## 1.1 [Sub-step]
## 1.2 [Sub-step]
### 1.2.1 [Sub-step]
# 2 [Main Step]
## 2.1 [Sub-step]
### 2.1.1 [Sub-step]
### 2.1.2 [Sub-step]
"""
    print(calculate_degree_and_depth(roadmap))
    