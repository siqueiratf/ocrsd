import json
import tempfile
import unittest
from pathlib import Path

from src.fields import (
    FieldDef, FieldElement, FieldMap, assign_components_to_elements,
    assign_components_to_fields, build_field_mask, load_field_map,
)


class FieldMapTests(unittest.TestCase):
    def test_loads_official_mapping_without_labels(self):
        from src.document_types import DOCUMENTS_ROOT

        path = DOCUMENTS_ROOT / "dnv/mappings/mapeamento_seguindo_norma.json"
        mapping = load_field_map(str(path))

        self.assertEqual(len(mapping.fields), 69)
        self.assertEqual(mapping.fields[0].label, "Nome do Recem-nascido (RN)")
        self.assertTrue(all(field.label for field in mapping.fields))

    def test_load_field_map_preserves_validation_metadata(self):
        payload = {
            "template_id": "sample",
            "dpi": 300,
            "page_size_px": [100, 200],
            "fields": [
                {
                    "field_id": "uf",
                    "label": "UF",
                    "bbox": [10, 20, 30, 40],
                    "data_type": "uf",
                    "validation": {"allowed_values": ["SP", "RJ"]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fields.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            field_map = load_field_map(str(path))

        field = field_map.fields[0]
        self.assertEqual(field.bbox, (10, 20, 30, 40))
        self.assertEqual(field.data_type, "uf")
        self.assertEqual(field.validation, {"allowed_values": ["SP", "RJ"]})

    def test_assigns_by_center_then_by_overlap(self):
        field_map = FieldMap(
            template_id="sample",
            dpi=300,
            page_size_px=(100, 100),
            fields=[
                FieldDef("left", "Left", (0, 0, 40, 40)),
                FieldDef("right", "Right", (60, 0, 100, 40)),
            ],
        )

        assigned = assign_components_to_fields(
            [(5, 5, 15, 15), (35, 10, 65, 20), (110, 10, 120, 20)], field_map
        )

        self.assertEqual(assigned["left"], [(5, 5, 15, 15), (35, 10, 65, 20)])
        self.assertEqual(assigned["right"], [])

    def test_compound_field_uses_and_assigns_element_boxes(self):
        field = FieldDef(
            "code", "Code", (0, 0, 100, 20), data_type="digit_sequence",
            elements=[
                FieldElement("d1", (10, 2, 20, 18), index=0),
                FieldElement("d2", (30, 2, 40, 18), index=1),
            ],
        )
        field_map = FieldMap("sample", 300, (100, 20), [field])

        mask = build_field_mask(field_map, (20, 100))
        assigned = assign_components_to_elements([(12, 4, 18, 14), (32, 4, 38, 14)], field)

        self.assertEqual(int(mask[10, 15]), 255)
        self.assertEqual(int(mask[10, 25]), 0)
        self.assertEqual(assigned["d1"], [(12, 4, 18, 14)])
        self.assertEqual(assigned["d2"], [(32, 4, 38, 14)])


if __name__ == "__main__":
    unittest.main()
