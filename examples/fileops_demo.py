#!/usr/bin/env python
"""Standalone demo for file operations tools.

This demo tests the file operations tools:
1. File manager (read, write, list, copy, move, delete)
2. Diff compare (unified, side-by-side, summary)

Usage:
    conda run -n agenticcli python examples/fileops_demo.py
"""

import sys
import tempfile
from pathlib import Path

from agentic_cli.tools.file_ops import file_manager, diff_compare


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

    result = file_manager(
        operation="write",
        path=str(test_file),
        content="Hello, World!\nThis is a test file.\nLine 3.",
    )

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Path: {result.get('path', 'N/A')}")
        print(f"    Size: {result.get('size', 'N/A')} bytes")
    print()

    return test_file


def demo_file_read(test_file: Path):
    """Demo file reading."""
    print("\n" + "=" * 60)
    print("File Read Demo")
    print("=" * 60)

    print(f"\n  Reading from: {test_file}")

    result = file_manager(
        operation="read",
        path=str(test_file),
    )

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Content:")
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

    result = file_manager(
        operation="list",
        path=str(temp_dir),
    )

    print(f"    Success: {result['success']}")
    if result['success']:
        print(f"    Entries:")
        for name, info in result['entries'].items():
            entry_type = info.get('type', 'unknown')
            type_icon = "D" if entry_type == 'directory' else "F"
            size = info.get('size', 0) or 0
            print(f"      [{type_icon}] {name:<15} ({size} bytes)")
    print()


def demo_file_copy(temp_dir: Path):
    """Demo file copying."""
    print("\n" + "=" * 60)
    print("File Copy Demo")
    print("=" * 60)

    source = temp_dir / "original.txt"
    dest = temp_dir / "copied.txt"

    # Create source file
    source.write_text("Original content to copy")

    print(f"\n  Source: {source.name}")
    print(f"  Destination: {dest.name}")

    result = file_manager(
        operation="copy",
        path=str(source),
        destination=str(dest),
    )

    print(f"    Success: {result['success']}")
    print(f"    Destination exists: {dest.exists()}")
    if dest.exists():
        print(f"    Content matches: {source.read_text() == dest.read_text()}")
    print()


def demo_file_move(temp_dir: Path):
    """Demo file moving/renaming."""
    print("\n" + "=" * 60)
    print("File Move Demo")
    print("=" * 60)

    source = temp_dir / "to_move.txt"
    dest = temp_dir / "moved.txt"

    # Create source file
    source.write_text("Content to move")

    print(f"\n  Source: {source.name}")
    print(f"  Destination: {dest.name}")

    result = file_manager(
        operation="move",
        path=str(source),
        destination=str(dest),
    )

    print(f"    Success: {result['success']}")
    print(f"    Source exists: {source.exists()}")
    print(f"    Destination exists: {dest.exists()}")
    print()


def demo_file_delete(temp_dir: Path):
    """Demo file deletion."""
    print("\n" + "=" * 60)
    print("File Delete Demo")
    print("=" * 60)

    file_to_delete = temp_dir / "deleteme.txt"
    file_to_delete.write_text("This will be deleted")

    print(f"\n  File to delete: {file_to_delete.name}")
    print(f"  Exists before: {file_to_delete.exists()}")

    result = file_manager(
        operation="delete",
        path=str(file_to_delete),
    )

    print(f"    Success: {result['success']}")
    print(f"    Exists after: {file_to_delete.exists()}")
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


def demo_error_handling(temp_dir: Path):
    """Demo error handling for file operations."""
    print("\n" + "=" * 60)
    print("Error Handling Demo")
    print("=" * 60)

    # Read non-existent file
    print("\n  Reading non-existent file:")
    result = file_manager(
        operation="read",
        path=str(temp_dir / "nonexistent.txt"),
    )
    print(f"    Success: {result['success']}")
    print(f"    Error: {result.get('error', 'N/A')}")

    # Invalid operation
    print("\n  Invalid operation:")
    result = file_manager(
        operation="invalid_op",
        path=str(temp_dir / "file.txt"),
    )
    print(f"    Success: {result['success']}")
    print(f"    Error: {result.get('error', 'N/A')}")

    # Diff with non-existent file (treated as raw text)
    print("\n  Comparing text with text:")
    result = diff_compare(
        source_a="Hello World",
        source_b="Hello Universe",
    )
    print(f"    Success: {result['success']}")
    print(f"    Similarity: {result.get('similarity', 0):.2%}")
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
        demo_file_list(temp_path)
        demo_file_copy(temp_path)
        demo_file_move(temp_path)
        demo_file_delete(temp_path)
        demo_diff_unified(temp_path)
        demo_diff_summary(temp_path)
        demo_diff_identical(temp_path)
        demo_error_handling(temp_path)

    print("\n" + "#" * 60)
    print("#  Demo Complete!")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
