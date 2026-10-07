import queue
import tkinter as tk
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.neural_handwriting import NeuralPen
from src.synthetic_dataset_gui import SyntheticDatasetFrame, _read_preferences


class SyntheticModelSelectionTests(unittest.TestCase):
    def test_select_all_fields_refreshes_preview(self):
        field_list = Mock()
        frame = SimpleNamespace(field_list=field_list)
        frame._render = Mock()

        SyntheticDatasetFrame._select_all_fields(frame)

        field_list.selection_set.assert_called_once_with(0, tk.END)
        frame._render.assert_called_once()

    def test_clear_field_selection_refreshes_preview(self):
        field_list = Mock()
        frame = SimpleNamespace(field_list=field_list)
        frame._render = Mock()

        SyntheticDatasetFrame._clear_field_selection(frame)

        field_list.selection_clear.assert_called_once_with(0, tk.END)
        frame._render.assert_called_once()

    def test_switching_models_updates_style_limits(self):
        interpreter = tk.Tcl()
        frame = SimpleNamespace(
            model_var=tk.StringVar(interpreter, "aaron1259"),
            writer_var=tk.StringVar(interpreter, "511"),
            bias_var=tk.DoubleVar(interpreter, 2.1),
            pen_width_var=tk.DoubleVar(interpreter, 7.0),
            pen_pressure_var=tk.DoubleVar(interpreter, 0.9),
            pen_dropout_var=tk.DoubleVar(interpreter, 0.8),
            text_policy_var=tk.StringVar(interpreter, "strict"),
            pen_color_var=tk.StringVar(interpreter, "#123456"),
            handwriting_var=tk.BooleanVar(interpreter, True),
            text_handwriting_var=tk.BooleanVar(interpreter, True),
            neural_controls=Mock(), writer_box=Mock(), handwriting_scale=Mock(),
            color_swatch=Mock(),
            model_preferences={
                "font": {"writer": "Aleatório", "bias": 0.75, "pen_width": 1.5, "pen_pressure": 0.25, "pen_dropout": 0.0, "text_policy": "strict"},
                "sjvasquez": {"writer": "Aleatório", "bias": 0.75, "pen_width": 1.5, "pen_pressure": 0.25, "pen_dropout": 0.0, "text_policy": "normalize"},
                "aaron1259": {"writer": "511", "bias": 2.1, "pen_width": 7.0, "pen_pressure": 0.9, "pen_dropout": 0.8, "text_policy": "strict"},
            },
            current_model="aaron1259",
            _loading_preferences=True,
        )
        frame._toggle_handwriting = lambda: SyntheticDatasetFrame._toggle_handwriting(frame)
        SyntheticDatasetFrame._model_changed(frame)
        self.assertEqual(len(frame.writer_box.configure.call_args.kwargs["values"]), 513)
        self.assertEqual(frame.bias_var.get(), 2.1)
        self.assertEqual(frame.writer_var.get(), "511")
        frame.model_var.set("sjvasquez")
        SyntheticDatasetFrame._model_changed(frame)
        self.assertEqual(len(frame.writer_box.configure.call_args.kwargs["values"]), 14)
        self.assertEqual(frame.bias_var.get(), 0.75)
        self.assertEqual(frame.pen_width_var.get(), 1.5)
        self.assertEqual(frame.pen_pressure_var.get(), 0.25)
        self.assertEqual(frame.pen_dropout_var.get(), 0.0)
        self.assertEqual(frame.text_policy_var.get(), "normalize")
        self.assertEqual(frame.writer_var.get(), "Aleatório")
        frame.model_var.set("font")
        SyntheticDatasetFrame._model_changed(frame)
        frame.neural_controls.grid_remove.assert_called_once()
        frame.handwriting_scale.configure.assert_called_with(state="normal")

    def test_preferences_are_saved_and_loaded(self):
        interpreter = tk.Tcl()
        with tempfile.TemporaryDirectory() as directory:
            prefs_path = Path(directory) / "prefs.json"
            frame = SimpleNamespace(
                count_var=tk.IntVar(interpreter, 25),
                seed_var=tk.StringVar(interpreter, "99"),
                handwriting_var=tk.BooleanVar(interpreter, False),
                text_handwriting_var=tk.BooleanVar(interpreter, True),
                handwriting_intensity_var=tk.DoubleVar(interpreter, 0.4),
                realistic_filters_var=tk.BooleanVar(interpreter, True),
                filter_preset_var=tk.StringVar(interpreter, "strong"),
                model_var=tk.StringVar(interpreter, "aaron1259"),
                writer_var=tk.StringVar(interpreter, "12"),
                bias_var=tk.DoubleVar(interpreter, 2.4),
                pen_color_var=tk.StringVar(interpreter, "#abcdef"),
                pen_width_var=tk.DoubleVar(interpreter, 4.5),
                pen_pressure_var=tk.DoubleVar(interpreter, 0.6),
                pen_dropout_var=tk.DoubleVar(interpreter, 0.2),
                text_policy_var=tk.StringVar(interpreter, "normalize"),
                model_preferences={
                    "font": {},
                    "sjvasquez": {},
                    "aaron1259": {},
                },
                _loading_preferences=False,
            )
            with patch("src.synthetic_dataset_gui.PREFERENCES_PATH", prefs_path):
                SyntheticDatasetFrame._save_preferences(frame)
                saved = _read_preferences(prefs_path)["synthetic_dataset"]

                loaded = SimpleNamespace(
                    model_preferences={
                        "font": {},
                        "sjvasquez": {},
                        "aaron1259": {},
                    },
                    count_var=tk.IntVar(interpreter),
                    seed_var=tk.StringVar(interpreter),
                    handwriting_var=tk.BooleanVar(interpreter),
                    text_handwriting_var=tk.BooleanVar(interpreter),
                    handwriting_intensity_var=tk.DoubleVar(interpreter),
                    realistic_filters_var=tk.BooleanVar(interpreter),
                    filter_preset_var=tk.StringVar(interpreter),
                    model_var=tk.StringVar(interpreter),
                    writer_var=tk.StringVar(interpreter),
                    bias_var=tk.DoubleVar(interpreter),
                    pen_color_var=tk.StringVar(interpreter),
                    pen_width_var=tk.DoubleVar(interpreter),
                    pen_pressure_var=tk.DoubleVar(interpreter),
                    pen_dropout_var=tk.DoubleVar(interpreter),
                    text_policy_var=tk.StringVar(interpreter),
                )
                SyntheticDatasetFrame._load_preferences(loaded)

        self.assertEqual(saved["model"], "aaron1259")
        self.assertEqual(saved["model_preferences"]["aaron1259"]["pen_width"], 4.5)
        self.assertTrue(saved["realistic_filters"])
        self.assertEqual(saved["filter_preset"], "strong")
        self.assertEqual(loaded.model_var.get(), "aaron1259")
        self.assertEqual(loaded.pen_width_var.get(), 4.5)
        self.assertEqual(loaded.writer_var.get(), "12")
        self.assertTrue(loaded.realistic_filters_var.get())
        self.assertEqual(loaded.filter_preset_var.get(), "strong")

    def test_worker_passes_model_settings_to_generator(self):
        frame = SimpleNamespace(events=queue.Queue())
        pen = NeuralPen(model="sjvasquez", writer=9, color="#123456", dropout=0.2)
        with patch("src.synthetic_dataset_gui.generate_dataset", return_value=[]) as generate:
            SyntheticDatasetFrame._worker(frame, Path("template"), Path("mapping"), {"name"},
                                          Path("output"), 1, 42, True, 0.65, True, pen, None, None, True, "light", "style_123")
        self.assertEqual(generate.call_args.kwargs["neural_pen"], pen)
        self.assertTrue(generate.call_args.kwargs["realistic_filters"])
        self.assertEqual(generate.call_args.kwargs["filter_preset"], "light")
        self.assertEqual(generate.call_args.kwargs["style_preset_uid"], "style_123")
        self.assertTrue(callable(generate.call_args.kwargs["progress_callback"]))
        self.assertEqual(frame.events.get_nowait(), ("log", "Carregando template e modelos..."))
        self.assertEqual(frame.events.get_nowait(), ("done", []))

    def test_refresh_presets_preserves_selected_uid(self):
        interpreter = tk.Tcl()
        frame = SimpleNamespace(
            style_preset_var=tk.StringVar(interpreter, "Caneta antiga · style_1"),
            number_style_var=tk.StringVar(interpreter, "Configuração atual"),
            text_style_var=tk.StringVar(interpreter, "Texto antigo · style_2"),
            style_preset_uids={
                "Configuração atual": None,
                "Caneta antiga · style_1": "style_1",
                "Texto antigo · style_2": "style_2",
            },
            style_preset_box=Mock(),
            number_style_box=Mock(),
            text_style_box=Mock(),
        )
        frame._label_for_uid = lambda uid: SyntheticDatasetFrame._label_for_uid(frame, uid)
        with patch("src.synthetic_dataset_gui.list_presets", return_value=[
            {"uid": "style_1", "name": "Caneta nova"},
            {"uid": "style_2", "name": "Texto novo"},
        ]):
            SyntheticDatasetFrame._refresh_preset_options(frame)

        self.assertEqual(frame.style_preset_var.get(), "Caneta nova · style_1")
        self.assertEqual(frame.text_style_var.get(), "Texto novo · style_2")
        self.assertEqual(frame.number_style_var.get(), "Configuração atual")

    def test_open_generated_in_viewer_loads_last_folder_and_selects_tab(self):
        viewer = Mock()
        notebook = Mock()
        frame = SimpleNamespace(
            last_generated_folder=Path("/tmp/generated"),
            generated_viewer=viewer,
            generated_viewer_notebook=notebook,
        )

        SyntheticDatasetFrame._open_generated_in_viewer(frame)

        viewer.open_generated_folder.assert_called_once_with(Path("/tmp/generated"))
        notebook.select.assert_called_once_with(viewer)


if __name__ == "__main__":
    unittest.main()
