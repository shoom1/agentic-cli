#!/usr/bin/env python
"""Standalone demo for file operation tools.

This demo tests the file operation tools:
1. write_file (create and overwrite files)
2. read_file (read with optional offset/limit)
3. list_dir (directory listing with metadata)
4. diff_compare (unified, summary, identical files)

Usage:
    conda run -n agenticcli python examples/fileops_demo.py
"""

import tempfile
from pathlib import Path

from agentic_cli.tools.file_read import read_file, diff_compare
from agentic_cli.tools.file_write import write_file
from agentic_cli.tools.glob_tool import list_dir


# =============================================================================
# Demo Functions
# =============================================================================


def demo_file_write(temp_dir: Path):
    """Demo file writing."""
    print("\n" + "=" * 60)
    print("File Write Demo")
    print("=" * 60)

    # Write a simple file
    test_file = temp_dir / "hello.txt"
    print(f"\n  Writing to: {test_file}")

    result = write_file(
        path=str(test_file),
        content="Hello, World!\nThis is a test file.\nLine 3.",
    )

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Path: {result.get('path', 'N/A')}")
        print(f"    Size: {result.get('size', 'N/A')} bytes")
        print(f"    Created: {result.get('created', 'N/A')}")
    print()

    return test_file


def demo_file_read(test_file: Path):
    """Demo file reading."""
    print("\n" + "=" * 60)
    print("File Read Demo")
    print("=" * 60)

    print(f"\n  Reading from: {test_file}")

    result = read_file(path=str(test_file))

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Content:")
        for line in result['content'].split('\n'):
            print(f"      {line}")
    print()


def demo_file_read_with_offset(temp_dir: Path):
    """Demo reading file with offset and limit."""
    print("\n" + "=" * 60)
    print("File Read with Offset/Limit Demo")
    print("=" * 60)

    # Create a file with numbered lines
    lines = "\n".join(f"Line {i}: content here" for i in range(1, 21))
    target = temp_dir / "numbered.txt"
    write_file(path=str(target), content=lines)

    print(f"\n  File: {target.name} (20 lines)")

    # Read lines 5-10
    print("  Reading lines 5-10 (offset=4, limit=6):")
    result = read_file(path=str(target), offset=4, limit=6)

    if result['success']:
        print(f"    Lines read: {result.get('lines_read', 'N/A')}")
        print(f"    Total lines: {result.get('total_lines', 'N/A')}")
        for line in result['content'].split('\n'):
            print(f"      {line}")
    print()


def demo_file_list(temp_dir: Path):
    """Demo directory listing."""
    print("\n" + "=" * 60)
    print("Directory List Demo")
    print("=" * 60)

    # Create some files first
    (temp_dir / "file1.txt").write_text("content1")
    (temp_dir / "file2.py").write_text("# python file")
    (temp_dir / "subdir").mkdir(exist_ok=True)
    (temp_dir / "subdir" / "nested.txt").write_text("nested content")

    print(f"\n  Listing: {temp_dir}")

    result = list_dir(path=str(temp_dir))

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Total entries: {result['total']}")
        print(f"    Directories:")
        for d in result['directories']:
            print(f"      [D] {d['name']}")
        print(f"    Files:")
        for f in result['files']:
            size = f.get('size') or 0
            print(f"      [F] {f['name']:<15} ({size} bytes)")
    print()


def demo_file_overwrite(temp_dir: Path):
    """Demo overwriting an existing file."""
    print("\n" + "=" * 60)
    print("File Overwrite Demo")
    print("=" * 60)

    target = temp_dir / "overwrite_test.txt"

    # Create initial file
    result1 = write_file(path=str(target), content="Original content")
    print(f"\n  Created: {target.name}")
    print(f"    Created (new file): {result1.get('created')}")

    # Overwrite
    result2 = write_file(path=str(target), content="Updated content")
    print(f"  Overwrote: {target.name}")
    print(f"    Created (new file): {result2.get('created')}")

    # Verify
    result3 = read_file(path=str(target))
    print(f"    Content after overwrite: {result3['content']}")
    print()


def demo_diff_unified(temp_dir: Path):
    """Demo unified diff format."""
    print("\n" + "=" * 60)
    print("Unified Diff Demo")
    print("=" * 60)

    # Create two files with differences
    file1 = temp_dir / "version1.txt"
    file2 = temp_dir / "version2.txt"

    file1.write_text("""Line 1: Hello
Line 2: World
Line 3: This is version 1
Line 4: Common line
Line 5: Another common line
""")

    file2.write_text("""Line 1: Hello
Line 2: Universe
Line 3: This is version 2
Line 4: Common line
Line 5: Another common line
Line 6: New line added
""")

    print(f"\n  Comparing: {file1.name} vs {file2.name}")
    print("  Mode: unified")

    result = diff_compare(
        source_a=str(file1),
        source_b=str(file2),
        mode="unified",
    )

    print(f"\n    Success: {result['success']}")
    print(f"    Similarity: {result.get('similarity', 0):.2%}")
    if result.get('diff'):
        print(f"\n    Diff output:")
        for line in result['diff'].split('\n')[:15]:
            print(f"      {line}")
    print()


def demo_diff_summary(temp_dir: Path):
    """Demo diff summary format."""
    print("\n" + "=" * 60)
    print("Diff Summary Demo")
    print("=" * 60)

    file1 = temp_dir / "old_version.py"
    file2 = temp_dir / "new_version.py"

    file1.write_text("""def hello():
    print("Hello")

def world():
    print("World")

def unused():
    pass
""")

    file2.write_text("""def hello():
    print("Hello, there!")

def world():
    print("World")

def new_function():
    return 42
""")

    print(f"\n  Comparing: {file1.name} vs {file2.name}")
    print("  Mode: summary")

    result = diff_compare(
        source_a=str(file1),
        source_b=str(file2),
        mode="summary",
    )

    print(f"\n    Success: {result['success']}")
    print(f"    Similarity: {result.get('similarity', 0):.2%}")
    if result.get('summary'):
        print(f"\n    Summary:")
        summary = result['summary']
        print(f"      Lines added: {summary.get('added', 0)}")
        print(f"      Lines removed: {summary.get('removed', 0)}")
        print(f"      Lines changed: {summary.get('changed', 0)}")
    print()


def demo_diff_identical(temp_dir: Path):
    """Demo comparing identical files."""
    print("\n" + "=" * 60)
    print("Identical Files Diff Demo")
    print("=" * 60)

    file1 = temp_dir / "same1.txt"
    file2 = temp_dir / "same2.txt"

    content = "Identical content\nLine 2\nLine 3\n"
    file1.write_text(content)
    file2.write_text(content)

    print(f"\n  Comparing: {file1.name} vs {file2.name}")

    result = diff_compare(
        source_a=str(file1),
        source_b=str(file2),
    )

    print(f"    Success: {result['success']}")
    print(f"    Similarity: {result.get('similarity', 0):.2%}")
    print(f"    Files identical: {result.get('similarity', 0) == 1.0}")
    print()


def demo_diff_raw_text():
    """Demo comparing raw text strings."""
    print("\n" + "=" * 60)
    print("Raw Text Diff Demo")
    print("=" * 60)

    print("\n  Comparing raw text strings:")
    result = diff_compare(
        source_a="Hello World",
        source_b="Hello Universe",
    )
    print(f"    Success: {result['success']}")
    print(f"    Similarity: {result.get('similarity', 0):.2%}")
    print()


def demo_error_handling(temp_dir: Path):
    """Demo error handling for file operations."""
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)

    # Read non-existent file
    print("\n  Reading non-existent file:")
    result = read_file(path=str(temp_dir / "nonexistent.txt"))
    print(f"    Success: {result['success']}")
    print(f"    Error: {result.get('error', 'N/A')}")
    print()


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#  File Operations Demo")
    print("#" * 60)

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n  Using temp directory: {temp_path}")

        # Run demos
        test_file = demo_file_write(temp_path)
        demo_file_read(test_file)
        demo_file_read_with_offset(temp_path)
        demo_file_list(temp_path)
        demo_file_overwrite(temp_path)
        demo_diff_unified(temp_path)
        demo_diff_summary(temp_path)
        demo_diff_identical(temp_path)
        demo_diff_raw_text()
        demo_error_handling(temp_path)

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
