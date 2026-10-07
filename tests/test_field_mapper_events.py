import unittest

import numpy as np

from src.field_mapper_gui import DATA_TYPES, FieldMapperFrame


class _Selection:
    def __init__(self, iid: str):
        self.iid = iid

    def selection(self):
        return (self.iid,)


class FieldMapperEventTests(unittest.TestCase):
    def test_text_sequence_is_available_as_a_field_type(self):
        self.assertIn("text_sequence", DATA_TYPES)

    def test_first_subelement_uses_parent_id_and_label(self):
        field = {"field_id": "data_nascimento", "label": "Data nascimento", "elements": []}

        identity = FieldMapperFrame._next_element_identity(field)

        self.assertEqual(identity, ("data_nascimento_0", "Data nascimento 0", 0))

    def test_next_subelement_increments_highest_parent_suffix(self):
        field = {
            "field_id": "data_nascimento", "label": "Data nascimento",
            "elements": [
                {"element_id": "data_nascimento_0"},
                {"element_id": "customizado"},
                {"element_id": "data_nascimento_2"},
            ],
        }

        identity = FieldMapperFrame._next_element_identity(field)

        self.assertEqual(identity, ("data_nascimento_3", "Data nascimento 3", 3))

    def test_move_handle_is_offset_from_bottom_right_corner(self):
        mapper = object.__new__(FieldMapperFrame)
        mapper.scale = 0.5

        self.assertEqual(mapper._move_handle_point([10, 20, 110, 220]), (65, 120))

    def test_move_handle_selects_specific_overlapping_field(self):
        mapper = object.__new__(FieldMapperFrame)
        mapper.scale = 1
        mapper.fields = [
            {"bbox": [0, 0, 100, 100]},
            {"bbox": [20, 20, 80, 80]},
        ]

        self.assertEqual(mapper._hit_move_handle(110, 110), (0, None))
        self.assertEqual(mapper._hit_move_handle(90, 90), (1, None))

    def test_build_field_map_preserves_printed_text_flag(self):
        mapper = object.__new__(FieldMapperFrame)
        mapper.template_image = np.zeros((120, 240, 3), dtype=np.uint8)
        mapper.template_id = "template"
        mapper.template_path = None
        mapper.dpi = 72
        mapper.fields = [{
            "field_id": "nome",
            "label": "Nome",
            "description": "",
            "bbox": [10, 20, 100, 40],
            "data_type": "text",
            "render_as_printed": True,
        }]

        field_map = mapper.build_field_map()

        self.assertTrue(field_map["fields"][0]["render_as_printed"])

    def test_repeated_parent_selection_does_not_reenter_selector(self):
        mapper = object.__new__(FieldMapperFrame)
        mapper._loading_editor = False
        mapper.selected_index = 2
        mapper.selected_element_index = None
        mapper.field_list = _Selection("field:2")
        calls = []
        mapper.select_field = calls.append

        mapper._list_select(None)

        self.assertEqual(calls, [])

    def test_repeated_element_selection_does_not_reenter_selector(self):
        mapper = object.__new__(FieldMapperFrame)
        mapper._loading_editor = False
        mapper.selected_index = 2
        mapper.selected_element_index = 4
        mapper.field_list = _Selection("element:2:4")
        calls = []
        mapper.select_element = lambda field, element: calls.append((field, element))

        mapper._list_select(None)

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
