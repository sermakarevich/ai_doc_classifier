import unittest

from doc_extractor.prompts import merge_prompt, vision_forecast_prompt


class TestVisionForecastPrompt(unittest.TestCase):
    maxDiff = None

    def test_mentions_page_count(self):
        prompt = vision_forecast_prompt(n_pages=7)
        self.assertIn("7 images", prompt)

    def test_contains_each_forecast_field(self):
        prompt = vision_forecast_prompt(n_pages=1)
        for field in (
            "sector_name",
            "revenue_now",
            "revenue_forecast",
            "year_now",
            "year_forecast",
            "cagr",
            "profit",
        ):
            self.assertIn(field, prompt, f"Field '{field}' not found in prompt")

    def test_output_format_json_structure(self):
        prompt = vision_forecast_prompt(n_pages=1)
        self.assertIn('"forecasts"', prompt)

    def test_instructs_null_for_missing(self):
        prompt = vision_forecast_prompt(n_pages=1)
        self.assertIn("null", prompt)

    def test_instructs_no_invention(self):
        prompt = vision_forecast_prompt(n_pages=1)
        self.assertIn("Do not invent numbers", prompt)


class TestMergePrompt(unittest.TestCase):
    def setUp(self):
        self.field_name = "revenue"
        self.field_description = "Annual revenue in dollars"
        self.variants = [
            "$29 trillion to $48 trillion",
            "29-48 trillion USD",
            "trillion-dollar economy",
            "null",
            "$50+ trillion",
        ]

    def test_contains_field_name(self):
        prompt = merge_prompt(self.field_name, self.field_description, self.variants)
        self.assertIn(self.field_name, prompt)

    def test_contains_field_description(self):
        prompt = merge_prompt(self.field_name, self.field_description, self.variants)
        self.assertIn(self.field_description, prompt)

    def test_contains_all_variants(self):
        prompt = merge_prompt(self.field_name, self.field_description, self.variants)
        for variant in self.variants:
            self.assertIn(variant, prompt)

    def test_numbered_variants(self):
        prompt = merge_prompt(self.field_name, self.field_description, self.variants)
        for i in range(len(self.variants)):
            self.assertIn(f"{i+1}.", prompt)

    def test_json_structure(self):
        prompt = merge_prompt(self.field_name, self.field_description, self.variants)
        self.assertIn('"groups"', prompt)
        self.assertIn('"canonical_value"', prompt)
        self.assertIn('"variants"', prompt)

    def test_empty_variants(self):
        prompt = merge_prompt(self.field_name, self.field_description, [])
        self.assertIn(self.field_name, prompt)
        self.assertIn('"groups"', prompt)


if __name__ == "__main__":
    unittest.main()
