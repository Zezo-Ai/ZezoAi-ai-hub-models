# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from enum import Enum


class TestStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


def write_junit_testcase(
    junit_xml_path: str,
    testsuite: str,
    name: str,
    classname: str,
    status: TestStatus,
    message: str = "",
    text: str = "",
) -> None:
    """
    Add a test case to a JUnit XML file, creating the file if it doesn't exist.

    Parameters
    ----------
    junit_xml_path
        Path to the JUnit XML file.
    testsuite
        Name of the test suite.
    name
        Name of the test case.
    classname
        Class name for the test case.
    status
        Test result status.
    message
        Message for non-passing test cases.
    text
        Detailed text content (e.g. full traceback) for non-passing test cases.
    """
    # Load existing file or create new root
    if os.path.exists(junit_xml_path) and os.path.getsize(junit_xml_path) > 0:
        tree = ET.parse(junit_xml_path)
        root = tree.getroot()
        if root.tag == "testsuites":
            suite = None
            for s in root.findall("testsuite"):
                if s.get("name") == testsuite:
                    suite = s
                    break
            if suite is None:
                suite = ET.SubElement(
                    root,
                    "testsuite",
                    name=testsuite,
                    tests="0",
                    failures="0",
                    errors="0",
                    skipped="0",
                    time="0.000",
                )
        elif root.tag == "testsuite":
            if root.get("name") == testsuite:
                suite = root
                new_root = ET.Element("testsuites")
                new_root.append(root)
                root = new_root
                tree = ET.ElementTree(root)
            else:
                new_root = ET.Element("testsuites")
                new_root.append(root)
                suite = ET.SubElement(
                    new_root,
                    "testsuite",
                    name=testsuite,
                    tests="0",
                    failures="0",
                    errors="0",
                    skipped="0",
                    time="0.000",
                )
                root = new_root
                tree = ET.ElementTree(root)
        else:
            raise ValueError(f"Unknown root element: {root.tag}")
    else:
        root = ET.Element("testsuites")
        suite = ET.SubElement(
            root,
            "testsuite",
            name=testsuite,
            tests="0",
            failures="0",
            errors="0",
            skipped="0",
            time="0.000",
        )
        tree = ET.ElementTree(root)

    # Check for existing testcase with the same name+classname
    existing = None
    for tc_el in suite.findall("testcase"):
        if tc_el.get("name") == name and tc_el.get("classname") == classname:
            existing = tc_el
            break

    if existing is not None:
        existing_failed = (
            existing.find("failure") is not None or existing.find("error") is not None
        )
        if status == TestStatus.PASSED or existing_failed:
            # Don't overwrite a failure with a pass, and don't duplicate a failure
            return
        # Replace existing pass with new failure
        suite.remove(existing)
        suite.set("tests", str(int(suite.get("tests", 0)) - 1))

    # Add test case
    tc = ET.SubElement(suite, "testcase", name=name, classname=classname, time="0.000")

    _STATUS_XML: dict[TestStatus, tuple[str, str]] = {
        TestStatus.FAILED: ("failure", "failures"),
        TestStatus.ERROR: ("error", "errors"),
        TestStatus.SKIPPED: ("skipped", "skipped"),
    }
    if status in _STATUS_XML:
        tag, counter = _STATUS_XML[status]
        el = ET.SubElement(tc, tag, message=message or f"Test {status.value}")
        if text:
            el.text = text
        suite.set(counter, str(int(suite.get(counter, 0)) + 1))

    # Update suite test count
    suite.set("tests", str(int(suite.get("tests", 0)) + 1))

    # Update root counters
    if root.tag == "testsuites":
        total_tests = sum(int(s.get("tests", 0)) for s in root.findall("testsuite"))
        total_failures = sum(
            int(s.get("failures", 0)) for s in root.findall("testsuite")
        )
        total_errors = sum(int(s.get("errors", 0)) for s in root.findall("testsuite"))
        total_skipped = sum(int(s.get("skipped", 0)) for s in root.findall("testsuite"))
        root.set("tests", str(total_tests))
        root.set("failures", str(total_failures))
        root.set("errors", str(total_errors))
        root.set("skipped", str(total_skipped))

    # Write
    os.makedirs(os.path.dirname(junit_xml_path) or ".", exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(junit_xml_path, encoding="unicode", xml_declaration=True)
