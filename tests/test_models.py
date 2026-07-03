import json
from pathlib import Path

from src.doc_extractor.models import (
    ExtractedField,
    FieldHit,
    FieldSpec,
    OutputSchema,
    SchemaExtraction,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_FILE = ROOT / "schema_sample.json"


class TestOutputSchema:
    def test_parses_schema_sample_json(self):
        data = SCHEMA_FILE.read_text()
        schema = OutputSchema.model_validate_json(data)
        assert schema.schema_id == "mckinsey_report"
        assert len(schema.fields) == 7
        assert schema.fields[0].name == "title"
        assert schema.fields[0].description == "Full title of the report"
        assert schema.fields[0].type == "string"

    def test_dump_and_reload(self):
        data = SCHEMA_FILE.read_text()
        schema = OutputSchema.model_validate_json(data)
        dumped = schema.model_dump_json()
        restored = OutputSchema.model_validate_json(dumped)
        assert restored.schema_id == schema.schema_id
        assert len(restored.fields) == len(schema.fields)
        assert restored.fields[0].name == schema.fields[0].name


class TestSchemaExtractionRoundtrip:
    def test_roundtrip(self):
        original = SchemaExtraction(
            fields=[
                FieldHit(name="title", value="The Next Big Arenas of Competition", grounding="page 1"),
                FieldHit(name="publisher", value=None),
            ]
        )
        dumped = original.model_dump_json()
        restored = SchemaExtraction.model_validate_json(dumped)
        assert len(restored.fields) == len(original.fields)
        assert restored.fields[0].name == "title"
        assert restored.fields[0].value == "The Next Big Arenas of Competition"
        assert restored.fields[1].value is None


class TestFieldSpecDefaults:
    def test_type_defaults_to_string(self):
        spec = FieldSpec(name="x", description="desc")
        assert spec.type == "string"

    def test_type_overridden(self):
        spec = FieldSpec(name="x", description="desc", type="integer")
        assert spec.type == "integer"


class TestExtractedFieldNoneValue:
    def test_accepts_none_value(self):
        ef = ExtractedField(
            name="test_field",
            value=None,
            count=0,
            total_calls=3,
            score=0.0,
            candidates=[],
        )
        assert ef.value is None

    def test_accepts_string_value(self):
        ef = ExtractedField(
            name="test_field",
            value="hello",
            count=2,
            total_calls=3,
            score=2 / 3,
            candidates=[],
        )
        assert ef.value == "hello"
