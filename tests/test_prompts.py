import unittest

from src.doc_extractor.models import FieldSpec, OutputSchema, ValueGroup
from src.doc_extractor.prompts import extraction_prompt, merge_prompt


class TestExtractionPrompt(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.schema = OutputSchema(
            schema_id="test-schema",
            fields=[
                FieldSpec(name="revenue", description="Annual revenue", type="number"),
                FieldSpec(name="is_public", description="Whether company is publicly traded", type="boolean"),
                FieldSpec(name="sectors", description="List of industry sectors", type="list"),
                FieldSpec(name="name", description="Company name", type="string"),
            ],
        )

    def test_contains_each_field_name(self):
        prompt = extraction_prompt("some doc", self.schema)
        for field in self.schema.fields:
            self.assertIn(field.name, prompt, f"Field '{field.name}' not found in prompt")

    def test_contains_each_description(self):
        prompt = extraction_prompt("some doc", self.schema)
        for field in self.schema.fields:
            self.assertIn(field.description, prompt, f"Description '{field.description}' not found in prompt")

    def test_truncates_long_doc_to_60000_chars(self):
        long_doc = "x" * 100_000
        prompt = extraction_prompt(long_doc, self.schema)
        # Per spec: doc_text truncated to 60_000 chars
        truncated_doc = long_doc[:60_000]
        self.assertIn(truncated_doc, prompt)
        # Total docpart (between "Document (truncated to 60000 chars):\n" and "Fields to extract:")
        doc_marker = "Document (truncated to 60000 chars):\n"
        fields_marker = "\nFields to extract:"
        doc_start = prompt.index(doc_marker) + len(doc_marker)
        fields_start = prompt.index(fields_marker)
        docpart = prompt[doc_start:fields_start]
        self.assertLessEqual(len(docpart), 61_000, f"docpart={len(docpart)}")

    def test_output_format_json_structure(self):
        prompt = extraction_prompt("test doc", self.schema)
        self.assertIn('"fields"', prompt)
        self.assertIn('"name"', prompt)
        self.assertIn('"value"', prompt)
        self.assertIn('"grounding"', prompt)

    def test_short_doc_unchanged(self):
        short_doc = "this is a short document"
        prompt = extraction_prompt(short_doc, self.schema)
        self.assertIn(short_doc, prompt)

    def test_includes_type(self):
        prompt = extraction_prompt("doc", self.schema)
        for field in self.schema.fields:
            self.assertIn(f"({field.type})", prompt)

    def test_json_structure(self):
        prompt = extraction_prompt("test", self.schema)
        self.assertIn('"fields"', prompt)


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
