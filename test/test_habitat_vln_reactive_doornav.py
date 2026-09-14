"""Compatibility discovery bridge for the documented focused-test command."""

import importlib
import unittest

TEST_MODULES = (
    "test_habitat_vln_reactive_doornav_contracts",
    "test_habitat_vln_reactive_doornav_no_privileged_leakage",
    "test_habitat_vln_reactive_doornav_runtime",
)


def load_tests(loader, tests, pattern):
    """Load the CI-named B1 suites without duplicating their assertions."""
    del tests, pattern
    suite = unittest.TestSuite()
    for module_name in TEST_MODULES:
        suite.addTests(
            loader.loadTestsFromModule(importlib.import_module(module_name))
        )
    return suite


if __name__ == "__main__":
    unittest.main()
