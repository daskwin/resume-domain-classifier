from __future__ import annotations

import importlib


def test_package_importable() -> None:
    module = importlib.import_module("resume_domain_classifier")
    assert module is not None


def test_commands_module_importable() -> None:
    module = importlib.import_module("resume_domain_classifier.commands")
    assert module is not None
