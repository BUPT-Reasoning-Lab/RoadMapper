import os
import json
import re


def extract_sections_from_md(md_content):
    """
    Extract all sections from markdown content.
    """
    # Use regex to match lines starting with #
    section_pattern = re.compile(r"^(#+\s+.+)$", re.MULTILINE)

    # Find all title positions
    section_matches = list(section_pattern.finditer(md_content))

    sections = []

    # Process each section
    for i, match in enumerate(section_matches):
        title = match.group(1).strip()
        start_pos = match.end()

        # If not the last section, content ends at the start of next section
        if i < len(section_matches) - 1:
            end_pos = section_matches[i + 1].start()
            content = md_content[start_pos:end_pos].strip()
        else:
            # Last section, content ends at file end
            content = md_content[start_pos:].strip()

        sections.append({"title": title, "content": content})

    return sections


def extract_content_before_title(md_content, target_title):
    """
    Extract all content before the specified title.

    Args:
        md_content: Markdown file content
        target_title: Target title (starting with #)

    Returns:
        Extracted content, or None if target title is not found
    """
    # Use regex to match lines starting with #
    section_pattern = re.compile(r"^(#+\s+.+)$", re.MULTILINE)

    # Find all title positions
    section_matches = list(section_pattern.finditer(md_content))

    # Find target title
    target_index = -1
    for i, match in enumerate(section_matches):
        title = match.group(1).strip()
        if title.startswith(target_title) or title == target_title:
            target_index = i
            break

    # If target title is found
    if target_index != -1:
        # If target title is the first title, return empty string
        if target_index == 0:
            return ""

        # Otherwise return all content from file start to before target title
        return md_content[: section_matches[target_index].start()].strip()

    return None


def process_md_files(input_dir, output_dir):
    """
    Process all md files in the specified directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all md files
    md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]

    # Statistics
    total_files = len(md_files)
    successful_files = 0
    failed_files = 0

    for md_file in md_files:
        input_path = os.path.join(input_dir, md_file)
        output_path = os.path.join(output_dir, md_file.replace(".md", ".json"))

        try:
            # Read md file content
            with open(input_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            # Extract sections
            sections = extract_sections_from_md(md_content)

            # Write to JSON file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)

            print(f"Processed: {md_file} -> {os.path.basename(output_path)}")
            successful_files += 1

        except Exception as e:
            print(f"Error processing file {md_file}: {str(e)}")
            failed_files += 1

    # Return statistics
    return {
        "total": total_files,
        "successful": successful_files,
        "failed": failed_files,
    }


def process_md_files_before_title(input_dir, output_dir, target_title):
    """
    Process all md files in the specified directory, extract content before the specified title.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        target_title: Target title
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all md files
    md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]

    # Statistics
    total_files = len(md_files)
    successful_files = 0
    failed_files = 0
    title_not_found = 0

    for md_file in md_files:
        input_path = os.path.join(input_dir, md_file)
        output_path = os.path.join(output_dir, md_file)

        try:
            # Read md file content
            with open(input_path, "r", encoding="utf-8") as f:
                md_content = f.read()

            # Extract content before specified title
            content_before_title = extract_content_before_title(
                md_content, target_title
            )

            # If target title is found
            if content_before_title is not None:
                # Write to output file
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content_before_title)

                print(f"Extracted content before {target_title}: {md_file}")
                successful_files += 1
            else:
                # Create empty file when title is not found
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("")

                print(f"Title {target_title} not found in file: {md_file}, created empty file")
                title_not_found += 1

        except Exception as e:
            print(f"Error processing file {md_file}: {str(e)}")
            failed_files += 1

    # Return statistics
    return {
        "total": total_files,
        "successful": successful_files,
        "failed": failed_files,
        "title_not_found": title_not_found,
    }


def main():
    # Use environment variables or default to relative paths
    input_dir = os.getenv("INPUT_DIR", "./data/input")
    output_dir = os.getenv("OUTPUT_DIR", "./data/output")

    # Specify target title and output folder
    target_title = os.getenv("TARGET_TITLE", "# Table of Contents")
    before_title_output_dir = os.getenv(
        "BEFORE_TITLE_OUTPUT_DIR", "./data/before-title"
    )

    # Process all files and generate JSON
    json_stats = process_md_files(input_dir, output_dir)
    print(f"All files processed. Results saved to: {output_dir}")
    print(
        f"JSON processing stats: Total: {json_stats['total']}, "
        f"Successful: {json_stats['successful']}, Failed: {json_stats['failed']}"
    )

    # Extract content before specified title
    before_title_stats = process_md_files_before_title(
        input_dir, before_title_output_dir, target_title
    )
    print(f"All files processed. Content before title saved to: {before_title_output_dir}")
    print(
        f"Title extraction stats: Total: {before_title_stats['total']}, "
        f"Successful: {before_title_stats['successful']}, "
        f"Failed: {before_title_stats['failed']}, "
        f"Title not found: {before_title_stats['title_not_found']}"
    )


if __name__ == "__main__":
    main()
