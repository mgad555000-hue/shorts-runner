import sys
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from review_evidence import (  # noqa: E402
    fit_overlong_evidence_quote,
    validate_evidence_quote,
)


class ReviewEvidenceTests(unittest.TestCase):
    def assert_valid(self, quote, source):
        self.assertEqual("", validate_evidence_quote(quote, [source]))

    def assert_invalid(self, quote, source):
        self.assertTrue(validate_evidence_quote(quote, [source]))

    def test_safe_arabic_orthographic_differences_are_accepted(self):
        source = "ومراقبة مستويات الفلترة أساسية لصحة الكلى"
        self.assert_valid(
            "مراقبه مستويات الفلتره",
            source,
        )
        self.assert_valid(
            "ومراقبه مستويات الفلتره",
            "مراقبة مستويات الفلترة أساسية لصحة الكلى",
        )
        self.assert_valid(
            "فمراقبه مستويات الفلتره",
            "مراقبة مستويات الفلترة أساسية لصحة الكلى",
        )
        self.assert_valid(
            "مُراقَبة مـستويات، الفلترة",
            source,
        )
        self.assert_valid(
            "اشارات مبكره للكلى",
            "إشارات مبكرة للكلي تستحق الانتباه",
        )

    def test_word_boundaries_and_internal_changes_are_rejected(self):
        self.assert_invalid(
            "ؤشرات أولية مهمة",
            "مؤشرات أولية مهمة تساعد على المتابعة",
        )
        self.assert_invalid(
            "مراقبة ومستويات الفلترة",
            "مراقبة مستويات الفلترة تساعد الطبيب",
        )
        self.assert_invalid(
            "بمراقبة مستويات الفلترة",
            "مراقبة مستويات الفلترة تساعد الطبيب",
        )
        self.assert_invalid(
            "مستويات مراقبة الفلترة",
            "مراقبة مستويات الفلترة تساعد الطبيب",
        )

    def test_word_count_is_checked_after_normalization(self):
        self.assert_invalid("كلمتان فقط", "كلمتان فقط هنا")
        thirteen = " ".join(f"كلمة{number}" for number in range(13))
        self.assert_invalid(thirteen, thirteen)
        self.assert_invalid("،،، ـــ", "أي مصدر صالح")
        self.assertTrue(validate_evidence_quote(123, ["أي مصدر صالح"]))

    def test_each_source_is_checked_separately(self):
        self.assertTrue(
            validate_evidence_quote(
                "نهاية المصدر بداية المصدر",
                ["هذه نهاية المصدر", "بداية المصدر هنا"],
            )
        )

    def test_medical_number_punctuation_is_not_erased(self):
        self.assert_invalid(
            "النسبة بين 1.5 و2",
            "النسبة بين 1-5 و2 تحتاج متابعة",
        )
        self.assert_valid(
            "النسبة بين 1.5 و2",
            "النسبة بين 1.5 و2 تحتاج متابعة",
        )

    def test_overlong_literal_quote_is_safely_shortened_to_source_prefix(self):
        source = ", ".join(
            f"معلومة صوديوم {number}" for number in range(1, 19)
        )
        fitted, error = fit_overlong_evidence_quote(source, [source])
        self.assertEqual("", error)
        self.assertEqual(12, len(fitted.replace(",", "").split()))
        self.assertEqual("", validate_evidence_quote(fitted, [source]))

    def test_overlong_nonliteral_quote_is_not_repaired(self):
        source = " ".join(f"كلمة{number}" for number in range(1, 20))
        alien = source + " عبارة دخيلة"
        fitted, error = fit_overlong_evidence_quote(alien, [source])
        self.assertEqual(alien, fitted)
        self.assertTrue(error)

    def test_valid_quote_is_not_changed(self):
        source = "مراقبة وظائف الكلى تساعد على فهم المؤشرات المبكرة"
        quote = "وظائف الكلى تساعد على فهم"
        fitted, error = fit_overlong_evidence_quote(quote, [source])
        self.assertEqual(quote, fitted)
        self.assertEqual("", error)


if __name__ == "__main__":
    unittest.main()
