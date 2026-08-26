"""The field registry must cover every Config option.

The GUI settings page is generated from `FIELD_SPECS`, so a field added to
`Config` without an entry here would silently never appear in the GUI. This
test is what makes that impossible.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autolock.config import (  # noqa: E402
    FIELD_SPECS,
    GROUP_ORDER,
    Config,
    coerce_value,
    grouped_fields,
)


class RegistryCoverageTests(unittest.TestCase):
    def test_every_config_field_is_in_the_registry(self):
        missing = {f.name for f in fields(Config)} - set(FIELD_SPECS)
        self.assertEqual(
            missing,
            set(),
            f"these settings would never appear in the GUI: {sorted(missing)}",
        )

    def test_the_registry_has_no_fields_that_do_not_exist(self):
        stale = set(FIELD_SPECS) - {f.name for f in fields(Config)}
        self.assertEqual(stale, set(), f"registry entries for removed fields: {sorted(stale)}")

    def test_every_group_is_known_and_ordered(self):
        for key, spec in FIELD_SPECS.items():
            self.assertIn(spec.group, GROUP_ORDER, f"{key} is in an unlisted group")

    def test_every_field_has_a_label_and_help(self):
        for key, spec in FIELD_SPECS.items():
            self.assertTrue(spec.label.strip(), f"{key} has no label")
            self.assertTrue(spec.help.strip(), f"{key} has no help text")
            self.assertNotEqual(spec.label, key, f"{key} needs a human label")

    def test_ranged_fields_have_a_sane_range(self):
        for key, spec in FIELD_SPECS.items():
            if spec.minimum is None:
                continue
            self.assertIsNotNone(spec.maximum, f"{key} has a minimum but no maximum")
            self.assertLess(spec.minimum, spec.maximum, f"{key} has an inverted range")

    def test_defaults_sit_inside_their_declared_range(self):
        defaults = Config()
        for key, spec in FIELD_SPECS.items():
            if spec.minimum is None:
                continue
            value = getattr(defaults, key)
            self.assertGreaterEqual(value, spec.minimum, f"{key} default is below its minimum")
            self.assertLessEqual(value, spec.maximum, f"{key} default is above its maximum")

    def test_choice_fields_default_to_one_of_their_choices(self):
        defaults = Config()
        for key, spec in FIELD_SPECS.items():
            if not spec.choices:
                continue
            self.assertIn(getattr(defaults, key), spec.choices, f"{key} default is not a choice")

    def test_ranges_are_only_declared_for_numbers(self):
        defaults = Config()
        for key, spec in FIELD_SPECS.items():
            if spec.minimum is None:
                continue
            value = getattr(defaults, key)
            self.assertIsInstance(value, (int, float), f"{key} has a range but is not numeric")
            self.assertNotIsInstance(value, bool, f"{key} is a boolean with a range")

    def test_grouped_fields_returns_everything_once(self):
        flat = [key for keys in grouped_fields().values() for key in keys]
        self.assertEqual(sorted(flat), sorted(FIELD_SPECS))
        self.assertEqual(len(flat), len(set(flat)), "a field appears in two groups")

    def test_hiding_advanced_fields_still_leaves_a_usable_page(self):
        basic = grouped_fields(include_advanced=False)
        flat = [key for keys in basic.values() for key in keys]
        self.assertLess(len(flat), len(FIELD_SPECS), "nothing is marked advanced")
        self.assertIn("absence_timeout_s", flat, "the headline setting must never be hidden")
        self.assertIn("lock_on_unknown", flat)


class RoundTripTests(unittest.TestCase):
    """What the GUI writes into a widget must survive coming back out."""

    def test_every_field_round_trips_through_its_string_form(self):
        cfg = Config()
        for key in FIELD_SPECS:
            value = getattr(cfg, key)
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, tuple):
                text = ",".join(str(part) for part in value)
            else:
                text = str(value)

            restored = coerce_value(cfg, key, text)
            self.assertEqual(restored, value, f"{key} did not survive a string round trip")
            self.assertIsInstance(restored, type(value), f"{key} changed type")

    def test_a_slider_float_lands_back_on_an_int_field(self):
        cfg = Config()
        self.assertEqual(coerce_value(cfg, "confirm_frames", "3.0"), 3)
        self.assertIsInstance(coerce_value(cfg, "confirm_frames", "3.0"), int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
