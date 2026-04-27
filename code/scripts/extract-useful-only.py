import os
import re


def split_blocks(md_content):
    """Split md content by titles, return [(title, content)]"""
    blocks = []
    titles = [m.start() for m in re.finditer(r"^# .+", md_content, re.MULTILINE)]
    titles.append(len(md_content))
    for i in range(len(titles) - 1):
        block = md_content[titles[i] : titles[i + 1]]
        lines = block.splitlines()
        if lines:
            title = lines[0].strip()
            content = "\n".join(lines[1:]).strip()
            blocks.append((title, content))
    return blocks


def get_titles_from_b(md_content):
    """Get all titles starting with # from file b"""
    return set(
        line.strip()
        for line in md_content.splitlines()
        if line.strip().startswith("# ")
    )


def remove_images(lines):
    # Remove image links like ![](images/xxx.jpg)
    img_pattern = re.compile(r"!\[.*?\]\(.*?\)")
    return [img_pattern.sub("", l) for l in lines]


def remove_tables(lines):
    # Remove HTML table elements <table>...</table> (can span multiple lines)
    new_lines = []
    in_table = False
    for l in lines:
        if "<table" in l:
            in_table = True
        if not in_table:
            new_lines.append(l)
        if "</table>" in l:
            in_table = False
    return new_lines


def remove_math(lines):
    # Remove formulas wrapped in $$...$$ (can span multiple lines)
    new_lines = []
    in_math = False
    for l in lines:
        if not in_math and "$$" in l:
            if l.count("$$") == 2:
                # Single-line formula, remove entire line
                continue
            else:
                in_math = True
                continue
        if in_math:
            if "$$" in l:
                in_math = False
            continue
        new_lines.append(l)
    return new_lines


def process_file(a_path, b_path, c_path):
    with open(a_path, "r", encoding="utf-8") as fa:
        a_content = fa.read()
    with open(b_path, "r", encoding="utf-8") as fb:
        b_content = fb.read()
    keep_titles = get_titles_from_b(b_content)
    blocks = split_blocks(a_content)
    result = []
    for title, content in blocks:
        if title in keep_titles:
            lines = [title]
            if content:
                lines += content.splitlines()
            # Remove images, tables, and formulas in sequence
            lines = remove_images(lines)
            lines = remove_tables(lines)
            lines = remove_math(lines)
            # Remove all blank lines
            lines = [l for l in lines if l.strip() != ""]
            if lines:
                result.append("\n".join(lines))
    # Remove all blank lines from final result
    final_text = "\n".join(
        [line for block in result for line in block.splitlines() if line.strip() != ""]
    )
    with open(c_path, "w", encoding="utf-8") as fc:
        fc.write(final_text)
    print(f"Processed: {a_path} -> {c_path}")


def main(dir_a, dir_b, dir_c):
    if not os.path.exists(dir_c):
        os.makedirs(dir_c)
    for filename in os.listdir(dir_a):
        if filename.endswith(".md"):
            a_path = os.path.join(dir_a, filename)
            b_path = os.path.join(dir_b, filename)
            c_path = os.path.join(dir_c, filename)
            if os.path.exists(b_path):
                process_file(a_path, b_path, c_path)
            else:
                print(f"Warning: {filename} not found in directory b, skipping.")


if __name__ == "__main__":
    # Use environment variables or default to relative paths
    dir_a = ""
    dir_b = ""
    dir_c = ""

    main(dir_a, dir_b, dir_c)
