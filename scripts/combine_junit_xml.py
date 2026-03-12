#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import glob
import os
import xml.etree.ElementTree as ET


def find_xml_files_recursive(directory: str, pattern: str = "*.xml") -> list[str]:
    """
    Recursively find all XML files in a directory.

    Parameters
    ----------
    directory
        Directory to search in.
    pattern
        Glob pattern to match (default: "*.xml").

    Returns
    -------
    file_paths : list[str]
        List of file paths.
    """
    return sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True))


def write_combined_xml(file_paths: list[str], output_path: str) -> None:
    """
    Combine multiple JUnit XML files into a single file.

    Parameters
    ----------
    file_paths
        List of paths to JUnit XML files to combine.
    output_path
        Path to write the combined JUnit XML file.
    """
    if not file_paths:
        print("No input files to combine")
        return

    # Collect all testsuites from all files
    all_testsuites: list[ET.Element] = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    for xml_file in file_paths:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Handle both <testsuites> and <testsuite> roots
            if root.tag == "testsuites":
                for testsuite in root.findall("testsuite"):
                    all_testsuites.append(testsuite)
                    total_tests += int(testsuite.get("tests", 0))
                    total_failures += int(testsuite.get("failures", 0))
                    total_errors += int(testsuite.get("errors", 0))
                    total_skipped += int(testsuite.get("skipped", 0))
                    total_time += float(testsuite.get("time", 0))
            elif root.tag == "testsuite":
                all_testsuites.append(root)
                total_tests += int(root.get("tests", 0))
                total_failures += int(root.get("failures", 0))
                total_errors += int(root.get("errors", 0))
                total_skipped += int(root.get("skipped", 0))
                total_time += float(root.get("time", 0))
            else:
                print(f"Warning: Unknown root element '{root.tag}' in {xml_file}")

        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {xml_file}: {e}")

    if not all_testsuites:
        print("No test suites found in any input files")
        return

    # Create combined root element
    combined_root = ET.Element("testsuites")
    combined_root.set("tests", str(total_tests))
    combined_root.set("failures", str(total_failures))
    combined_root.set("errors", str(total_errors))
    combined_root.set("skipped", str(total_skipped))
    combined_root.set("time", f"{total_time:.3f}")

    # Add all testsuites to combined root
    for testsuite in all_testsuites:
        combined_root.append(testsuite)

    # Write combined XML
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tree = ET.ElementTree(combined_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"Combined {len(file_paths)} files into {output_path}")
    print(f"  Total tests: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Time: {total_time:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple JUnit XML files into a single file"
    )
    parser.add_argument(
        "--junit-xml",
        "-j",
        required=True,
        help="Directory containing JUnit XML files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for combined JUnit XML file",
    )
    args = parser.parse_args()

    # Find all XML files recursively in input directory
    if not os.path.exists(args.junit_xml):
        print(f"Error: Input directory does not exist: {args.junit_xml}")
        return

    all_xml_files = find_xml_files_recursive(args.junit_xml)
    if not all_xml_files:
        print(f"No XML files found in {args.junit_xml}")
        return

    print(f"Found {len(all_xml_files)} XML files in {args.junit_xml}")

    # Write combined XML
    write_combined_xml(all_xml_files, args.output)


if __name__ == "__main__":
    main()
