import json
import tempfile
import unittest
from pathlib import Path

from src.writing_presets import (
    PRESET_LIBRARY_PATH, document_id_with_preset, find_preset, generation_settings, list_presets,
    suggested_preset_name, uid_from_pdf_name, upsert_preset,
)


class WritingPresetTests(unittest.TestCase):
    def test_default_library_path_is_project_documents_folder(self):
        self.assertEqual(PRESET_LIBRARY_PATH.name, "writing_presets.json")
        self.assertEqual(PRESET_LIBRARY_PATH.parent.name, "documents")
        self.assertTrue((PRESET_LIBRARY_PATH.parents[1] / "src").is_dir())

    def test_uid_is_appended_and_parsed_from_pdf_name(self):
        document_id = document_id_with_preset("synthetic_0001", "style_abc123")

        self.assertEqual(document_id, "synthetic_0001__style-style_abc123")
        self.assertEqual(uid_from_pdf_name(f"{document_id}.pdf"), "style_abc123")
        self.assertIsNone(uid_from_pdf_name("synthetic_0001.pdf"))

    def test_presets_are_stored_as_a_json_list(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writing_presets.json"
            preset = upsert_preset({
                "uid": "style_test",
                "name": "Caneta azul",
                "settings": generation_settings(
                    model="font", writer="Aleatório", bias=0.75,
                    pen_color="#193787", pen_width=1.4,
                    pen_pressure=0.2, pen_dropout=0.0,
                    text_policy="strict",
                ),
            }, path)

            raw = json.loads(path.read_text(encoding="utf-8"))
            listed = list_presets(path)
            found = find_preset("style_test", path)

            self.assertEqual(raw["presets"][0]["uid"], "style_test")
            self.assertEqual(listed[0], preset)
            self.assertEqual(found, preset)

    def test_suggests_descriptive_preset_name(self):
        settings = generation_settings(
            model="aaron1259", writer="12", bias=2.1,
            pen_color="#193787", pen_width=2.25,
            pen_pressure=0.6, pen_dropout=0.05,
            text_policy="normalize", realistic_filters=True, filter_preset="strong",
        )

        name = suggested_preset_name(settings)

        self.assertEqual(
            name,
            "aaron1259_estilo-12_reg-2p1_cor-193787_esp-2p25_press-0p6_falha-0p05_normalize_filtro-strong",
        )

    def test_suggests_mixed_preset_name(self):
        number_settings = generation_settings(
            model="sjvasquez", writer="3", bias=0.75,
            pen_color="#193787", pen_width=1.5,
            pen_pressure=0.25, pen_dropout=0,
            text_policy="normalize",
        )
        text_settings = generation_settings(
            model="wpvnteam", writer="Aleatório", bias=2.1,
            pen_color="#111111", pen_width=2,
            pen_pressure=0.4, pen_dropout=0.1,
            text_policy="normalize",
        )

        name = suggested_preset_name({
            "mode": "mixed",
            "numbers": number_settings,
            "text": text_settings,
            "realistic_filters": False,
        })

        self.assertTrue(name.startswith("mixed_num_sjvasquez_estilo-3_reg-0p75"))
        self.assertIn("_txt_wpvnteam_estilo-aleatorio_reg-2p1_", name)
        self.assertTrue(name.endswith("_sem-filtro"))


if __name__ == "__main__":
    unittest.main()
