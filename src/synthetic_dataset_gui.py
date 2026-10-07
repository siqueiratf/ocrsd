from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from src.document_types import DocumentType, discover_document_types
from src.document_types import build_dataset_run_dir, infer_document_type_key
from src.pdf import pdf_to_image
from src.synthetic_dataset import generate_dataset
from src.neural_handwriting import NeuralPen
from src.writing_presets import (
    find_preset, generation_settings, list_presets, new_preset_uid,
    suggested_preset_name, upsert_preset,
)


PREFERENCES_PATH = Path.cwd() / ".ocrsd_gui_preferences.json"
PREFERENCES_KEY = "synthetic_dataset"
PEN_WIDTH_MAX = 40.0
MODEL_PROFILES = {
    "font": {
        "bias": 0.75,
        "pen_width": 1.5,
        "pen_pressure": 0.25,
        "pen_dropout": 0.0,
        "text_policy": "strict",
        "writer": "Aleatório",
    },
    "sjvasquez": {
        "bias": 0.75,
        "pen_width": 1.5,
        "pen_pressure": 0.25,
        "pen_dropout": 0.0,
        "text_policy": "normalize",
        "writer": "Aleatório",
    },
    "aaron1259": {
        "bias": 2.1,
        "pen_width": 1.5,
        "pen_pressure": 0.25,
        "pen_dropout": 0.0,
        "text_policy": "normalize",
        "writer": "Aleatório",
    },
    "wpvnteam": {
        "bias": 2.1,
        "pen_width": 1.5,
        "pen_pressure": 0.25,
        "pen_dropout": 0.0,
        "text_policy": "normalize",
        "writer": "Aleatório",
    },
}
MODEL_WRITER_MAX = {"sjvasquez": 12, "aaron1259": 511, "wpvnteam": 511}


def _read_preferences(path: Path | None = None) -> dict:
    path = path or PREFERENCES_PATH
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_preferences(preferences: dict, path: Path | None = None) -> None:
    path = path or PREFERENCES_PATH
    path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clamp_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, number))


def _variable_value(variable: tk.Variable, default: object) -> object:
    try:
        return variable.get()
    except tk.TclError:
        return default


def _valid_writer(value: object, model: str) -> str:
    if value in (None, "", "Aleatório"):
        return "Aleatório"
    try:
        writer = int(value)
    except (TypeError, ValueError):
        return "Aleatório"
    maximum = MODEL_WRITER_MAX.get(model)
    if maximum is None or not 0 <= writer <= maximum:
        return "Aleatório"
    return str(writer)


class SyntheticDatasetFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.document_types = discover_document_types()
        self._loading_preferences = True
        self.model_preferences = {model: dict(profile) for model, profile in MODEL_PROFILES.items()}
        self.type_var = tk.StringVar()
        self.template_var = tk.StringVar()
        self.mapping_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.count_var = tk.IntVar(value=10)
        self.seed_var = tk.StringVar(value="42")
        self.handwriting_var = tk.BooleanVar(value=True)
        self.text_handwriting_var = tk.BooleanVar(value=True)
        self.handwriting_intensity_var = tk.DoubleVar(value=0.65)
        self.realistic_filters_var = tk.BooleanVar(value=False)
        self.filter_preset_var = tk.StringVar(value="medium")
        self.model_var = tk.StringVar(value="font")
        self.style_preset_var = tk.StringVar(value="Configuração atual")
        self.split_style_var = tk.BooleanVar(value=False)
        self.number_style_var = tk.StringVar(value="Configuração atual")
        self.text_style_var = tk.StringVar(value="Configuração atual")
        self.style_preset_uids: dict[str, str | None] = {"Configuração atual": None}
        self.writer_var = tk.StringVar(value="Aleatório")
        self.bias_var = tk.DoubleVar(value=0.75)
        self.pen_color_var = tk.StringVar(value="#193787")
        self.pen_width_var = tk.DoubleVar(value=1.5)
        self.pen_pressure_var = tk.DoubleVar(value=0.25)
        self.pen_dropout_var = tk.DoubleVar(value=0.0)
        self.text_policy_var = tk.StringVar(value="strict")
        self.status_var = tk.StringVar(value="Escolha o template e os campos que deseja preencher.")
        self.fields: list[dict] = []
        self.mapping_dpi = 300
        self.preview_image = None
        self.preview_photo = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.generated_viewer = None
        self.generated_viewer_notebook = None
        self.last_generated_folder: Path | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._load_preferences()
        self.current_model = self.model_var.get()
        self._build_ui()
        self._bind_preference_traces()
        self._loading_preferences = False
        self._apply_model_profile(self.model_var.get(), reset=False)
        self._refresh_preset_options()
        self.after(100, self._poll_events)
        if self.document_types:
            first = next(iter(self.document_types.values()))
            self.type_var.set(first.label)
            self._select_type()

    def set_generated_pdf_viewer(self, viewer: object, notebook: object | None = None) -> None:
        self.generated_viewer = viewer
        self.generated_viewer_notebook = notebook

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Panel.TFrame", padding=10)
        header.pack(fill="x", pady=(10, 8))
        ttk.Label(header, text="Datasets artificiais", style="Panel.TLabel", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var, style="Panel.TLabel", anchor="center").pack(side="left", fill="x", expand=True, padx=12)
        self.open_viewer_button = ttk.Button(
            header, text="Abrir no visualizador", style="Accent.TButton",
            command=self._open_generated_in_viewer,
        )

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        viewer = ttk.Frame(body)
        controls = ttk.Frame(body, style="Panel.TFrame", width=380)
        body.add(viewer, weight=4)
        body.add(controls, weight=1)

        self.canvas = tk.Canvas(viewer, bg="#0e1012", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render())

        scroll_canvas = tk.Canvas(controls, bg="#202328", highlightthickness=0, width=350)
        scrollbar = ttk.Scrollbar(controls, orient="vertical", command=scroll_canvas.yview)
        panel = ttk.Frame(scroll_canvas, style="Panel.TFrame", padding=12)
        window = scroll_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _event: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.bind("<Configure>", lambda event: scroll_canvas.itemconfigure(window, width=event.width))
        def scroll_controls(event: tk.Event) -> str | None:
            left, top = controls.winfo_rootx(), controls.winfo_rooty()
            if not (
                left <= event.x_root < left + controls.winfo_width()
                and top <= event.y_root < top + controls.winfo_height()
            ):
                return None
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
            scroll_canvas.yview_scroll(direction, "units")
            return "break"
        scroll_canvas.bind_all("<MouseWheel>", scroll_controls, add="+")
        scroll_canvas.bind_all("<Button-4>", scroll_controls, add="+")
        scroll_canvas.bind_all("<Button-5>", scroll_controls, add="+")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(panel, text="1. Documento base", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        values = tuple(item.label for item in self.document_types.values())
        self.type_box = ttk.Combobox(panel, textvariable=self.type_var, values=values, state="readonly")
        self.type_box.pack(fill="x", pady=(5, 3))
        self.type_box.bind("<<ComboboxSelected>>", lambda _event: self._select_type())
        ttk.Button(panel, text="Abrir template personalizado…", command=self._browse_template).pack(fill="x", pady=3)
        ttk.Button(panel, text="Abrir mapeamento personalizado…", command=self._browse_mapping).pack(fill="x", pady=3)
        self.source_label = ttk.Label(panel, text="", style="Panel.TLabel", wraplength=325)
        self.source_label.pack(fill="x", pady=(4, 12))

        ttk.Label(panel, text="2. Campos sintéticos", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        self.field_list = tk.Listbox(
            panel, selectmode="extended", exportselection=False, height=14,
            bg="#292d33", fg="#f2f3f5", selectbackground="#34446f", relief="flat",
        )
        self.field_list.pack(fill="x", pady=(5, 3))
        self.field_list.bind("<<ListboxSelect>>", lambda _event: self._render())
        field_actions = ttk.Frame(panel, style="Panel.TFrame")
        field_actions.pack(fill="x", pady=(0, 12))
        ttk.Button(field_actions, text="Selecionar todos", command=self._select_all_fields).pack(side="left")
        ttk.Button(field_actions, text="Limpar", command=self._clear_field_selection).pack(side="right")

        ttk.Label(panel, text="3. Geração", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
        form = ttk.Frame(panel, style="Panel.TFrame")
        form.pack(fill="x", pady=5)
        ttk.Label(form, text="Quantidade", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=10000, textvariable=self.count_var, width=10).grid(row=0, column=1, sticky="e")
        ttk.Label(form, text="Seed", style="Panel.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.seed_var, width=12).grid(row=1, column=1, sticky="e", pady=5)
        ttk.Checkbutton(
            form, text="Números manuscritos", variable=self.handwriting_var,
            command=self._toggle_handwriting,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 2))
        ttk.Checkbutton(
            form, text="Textos manuscritos", variable=self.text_handwriting_var,
            command=self._toggle_handwriting,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(
            form, text="Aplicar filtros automaticamente", variable=self.realistic_filters_var,
            command=self._save_preferences,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Label(form, text="Intensidade dos filtros", style="Panel.TLabel").grid(row=5, column=0, sticky="w")
        ttk.Combobox(
            form, textvariable=self.filter_preset_var,
            values=("light", "medium", "strong"), state="readonly", width=10,
        ).grid(row=5, column=1, sticky="e", pady=3)
        ttk.Label(form, text="Variação da escrita", style="Panel.TLabel").grid(row=6, column=0, sticky="w")
        self.handwriting_scale = ttk.Scale(
            form, from_=0.0, to=1.0, variable=self.handwriting_intensity_var,
        )
        self.handwriting_scale.grid(row=6, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(form, text="Modelo de escrita", style="Panel.TLabel").grid(row=7, column=0, sticky="w", pady=5)
        model_box = ttk.Combobox(form, textvariable=self.model_var, values=("font", "sjvasquez", "aaron1259", "wpvnteam"), state="readonly", width=13)
        model_box.grid(row=7, column=1, sticky="ew")
        model_box.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(form, text="Preset de escrita", style="Panel.TLabel").grid(row=8, column=0, sticky="w", pady=5)
        self.style_preset_box = ttk.Combobox(form, textvariable=self.style_preset_var, state="readonly", width=18)
        self.style_preset_box.grid(row=8, column=1, sticky="ew")
        self.style_preset_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_selected_preset())
        preset_actions = ttk.Frame(form, style="Panel.TFrame")
        preset_actions.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ttk.Button(preset_actions, text="Atualizar presets", command=self._refresh_preset_options).pack(side="left")
        ttk.Button(preset_actions, text="Salvar atual", command=self._save_current_preset).pack(side="right")
        ttk.Checkbutton(
            form, text="Usar presets separados para números e textos",
            variable=self.split_style_var, command=self._toggle_split_presets,
        ).grid(row=10, column=0, columnspan=2, sticky="w", pady=(2, 3))
        self.split_controls = ttk.Frame(form, style="Panel.TFrame")
        self.split_controls.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.split_controls.columnconfigure(1, weight=1)
        ttk.Label(self.split_controls, text="Números", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.number_style_box = ttk.Combobox(self.split_controls, textvariable=self.number_style_var, state="readonly", width=18)
        self.number_style_box.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(self.split_controls, text="Textos", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        self.text_style_box = ttk.Combobox(self.split_controls, textvariable=self.text_style_var, state="readonly", width=18)
        self.text_style_box.grid(row=1, column=1, sticky="ew", pady=2)
        self.neural_controls = ttk.Frame(form, style="Panel.TFrame")
        self.neural_controls.grid(row=12, column=0, columnspan=2, sticky="ew")
        self.neural_controls.columnconfigure(1, weight=1)
        ttk.Label(self.neural_controls, text="Estilo", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        self.writer_box = ttk.Combobox(self.neural_controls, textvariable=self.writer_var, state="readonly", width=13)
        self.writer_box.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(self.neural_controls, text="Regularidade", style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(self.neural_controls, from_=0, to=3, increment=0.05,
                    textvariable=self.bias_var, width=10).grid(row=1, column=1, sticky="e", pady=3)
        ttk.Label(self.neural_controls, text="Espessura (px)", style="Panel.TLabel").grid(row=2, column=0, sticky="w")
        pen_width_value = ttk.Label(
            self.neural_controls, text=f"{float(self.pen_width_var.get()):.2f}",
            style="Panel.TLabel", width=5, anchor="e",
        )
        pen_width_value.grid(row=2, column=1, sticky="e", pady=3)
        ttk.Scale(
            self.neural_controls, from_=0.25, to=PEN_WIDTH_MAX, variable=self.pen_width_var,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        self.pen_width_var.trace_add(
            "write",
            lambda *_args, shown=pen_width_value: shown.configure(text=f"{float(self.pen_width_var.get()):.2f}"),
        )
        for row, (label, variable) in enumerate((
            ("Variação de pressão", self.pen_pressure_var),
            ("Falhas da caneta", self.pen_dropout_var),
        ), start=4):
            ttk.Label(self.neural_controls, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky="w")
            ttk.Spinbox(self.neural_controls, from_=0, to=1, increment=0.05,
                        textvariable=variable, width=10).grid(row=row, column=1, sticky="e", pady=3)
        ttk.Label(self.neural_controls, text="Cor da caneta", style="Panel.TLabel").grid(row=6, column=0, sticky="w")
        self.color_swatch = tk.Button(self.neural_controls, bg=self.pen_color_var.get(), width=4, command=self._choose_pen_color)
        self.color_swatch.grid(row=6, column=1, sticky="e", pady=3)
        ttk.Label(self.neural_controls, text="Caracteres não suportados", style="Panel.TLabel").grid(row=7, column=0, sticky="w")
        ttk.Combobox(self.neural_controls, textvariable=self.text_policy_var,
                     values=("strict", "normalize"), state="readonly", width=10).grid(row=7, column=1, sticky="e")
        self._model_changed()
        self._toggle_split_presets()
        form.columnconfigure(0, weight=1)
        self._toggle_handwriting()
        self.output_label = ttk.Label(panel, text="Saída automática em documents/<tipo>/dataset", style="Panel.TLabel", wraplength=325)
        self.output_label.pack(fill="x", pady=(2, 8))
        self.generate_button = ttk.Button(panel, text="Gerar dataset", style="Accent.TButton", command=self._generate)
        self.generate_button.pack(fill="x", pady=3)
        self.progress = ttk.Progressbar(panel, mode="indeterminate")
        self.progress.pack(fill="x", pady=(5, 0))
        ttk.Label(panel, text="Log da geração", style="Panel.TLabel", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(10, 3))
        self.generation_log = tk.Listbox(
            panel, height=6, bg="#292d33", fg="#f2f3f5",
            selectbackground="#34446f", relief="flat",
        )
        self.generation_log.pack(fill="x", pady=(0, 2))

    def _current_type(self) -> DocumentType | None:
        return next((item for item in self.document_types.values() if item.label == self.type_var.get()), None)

    def _select_type(self) -> None:
        item = self._current_type()
        if item is None:
            return
        self.template_var.set(str(item.template))
        if item.mappings:
            self.mapping_var.set(str(item.mappings[0]))
        self._refresh_output_preview()
        self.output_label.configure(text=self.output_var.get())
        self._load_sources()

    def _browse_template(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Template PDF", "*.pdf")])
        if path:
            self.template_var.set(path)
            self._load_sources()

    def _browse_mapping(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Mapeamento JSON", "*.json")])
        if path:
            self.mapping_var.set(path)
            self._load_sources()

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)
            self.output_label.configure(text=path)

    def _refresh_output_preview(self) -> None:
        try:
            count = int(self.count_var.get())
        except (ValueError, tk.TclError):
            count = 0
        template = Path(self.template_var.get()) if self.template_var.get() else None
        mapping = Path(self.mapping_var.get()) if self.mapping_var.get() else None
        document_type = infer_document_type_key(template, mapping)
        self.output_var.set(str(build_dataset_run_dir(document_type, max(1, count))))

    def _load_sources(self) -> None:
        template, mapping = Path(self.template_var.get()), Path(self.mapping_var.get())
        if not template.is_file() or not mapping.is_file():
            return
        try:
            data = json.loads(mapping.read_text(encoding="utf-8"))
            self.mapping_dpi = int(data.get("dpi", 300))
            self.fields = data.get("fields", [])
            self.field_list.delete(0, "end")
            for field in self.fields:
                self.field_list.insert("end", f"{field['field_id']} · {field.get('label', '')}")
            self.field_list.selection_set(0, "end")
            self.preview_image = pdf_to_image(str(template), 120).image
            self.source_label.configure(text=f"{template.name}\n{mapping.name} · {len(self.fields)} campos")
            self._refresh_output_preview()
            self.output_label.configure(text=self.output_var.get())
            self._render()
        except Exception as exc:
            messagebox.showerror("Dataset artificial", str(exc))

    def _selected_ids(self) -> set[str]:
        return {str(self.fields[index]["field_id"]) for index in self.field_list.curselection()}

    def _select_all_fields(self) -> None:
        self.field_list.selection_set(0, tk.END)
        self._render()

    def _clear_field_selection(self) -> None:
        self.field_list.selection_clear(0, tk.END)
        self._render()

    def _load_preferences(self) -> None:
        preferences = _read_preferences().get(PREFERENCES_KEY, {})
        model_preferences = preferences.get("model_preferences", {})
        for model, defaults in MODEL_PROFILES.items():
            saved = model_preferences.get(model, {})
            profile = {**defaults, **saved}
            profile["bias"] = _clamp_float(profile.get("bias"), defaults["bias"], 0, 3)
            profile["pen_width"] = _clamp_float(profile.get("pen_width"), defaults["pen_width"], 0.25, PEN_WIDTH_MAX)
            profile["pen_pressure"] = _clamp_float(profile.get("pen_pressure"), defaults["pen_pressure"], 0, 1)
            profile["pen_dropout"] = _clamp_float(profile.get("pen_dropout"), defaults["pen_dropout"], 0, 1)
            profile["text_policy"] = profile.get("text_policy") if profile.get("text_policy") in {"strict", "normalize"} else defaults["text_policy"]
            profile["writer"] = _valid_writer(profile.get("writer"), model)
            self.model_preferences[model] = profile
        model = preferences.get("model", "font")
        self.model_var.set(model if model in MODEL_PROFILES else "font")
        self.count_var.set(int(_clamp_float(preferences.get("count"), 10, 1, 10000)))
        self.seed_var.set(str(preferences.get("seed", "42")))
        self.handwriting_var.set(bool(preferences.get("handwriting", True)))
        self.text_handwriting_var.set(bool(preferences.get("text_handwriting", True)))
        self.handwriting_intensity_var.set(_clamp_float(preferences.get("handwriting_intensity"), 0.65, 0, 1))
        self.realistic_filters_var.set(bool(preferences.get("realistic_filters", False)))
        preset = str(preferences.get("filter_preset", "medium"))
        self.filter_preset_var.set(preset if preset in {"light", "medium", "strong"} else "medium")
        color = str(preferences.get("pen_color", "#193787"))
        self.pen_color_var.set(color if len(color) == 7 and color.startswith("#") else "#193787")
        SyntheticDatasetFrame._apply_model_profile(self, self.model_var.get(), reset=False, configure_widgets=False)

    def _bind_preference_traces(self) -> None:
        for variable in (
            self.count_var, self.seed_var, self.handwriting_var, self.text_handwriting_var,
            self.handwriting_intensity_var, self.realistic_filters_var, self.filter_preset_var,
            self.writer_var, self.bias_var,
            self.pen_color_var, self.pen_width_var, self.pen_pressure_var,
            self.pen_dropout_var, self.text_policy_var,
        ):
            variable.trace_add("write", lambda *_args: self._save_preferences())
        self.count_var.trace_add("write", lambda *_args: self._refresh_output_preview())

    def _capture_model_profile(self, model: str | None = None) -> None:
        model = model or self.model_var.get()
        if model not in MODEL_PROFILES:
            return
        defaults = MODEL_PROFILES[model]
        text_policy = _variable_value(self.text_policy_var, defaults["text_policy"])
        self.model_preferences[model] = {
            "writer": _valid_writer(_variable_value(self.writer_var, "Aleatório"), model),
            "bias": _clamp_float(_variable_value(self.bias_var, defaults["bias"]), defaults["bias"], 0, 3),
            "pen_width": _clamp_float(_variable_value(self.pen_width_var, defaults["pen_width"]), defaults["pen_width"], 0.25, PEN_WIDTH_MAX),
            "pen_pressure": _clamp_float(_variable_value(self.pen_pressure_var, defaults["pen_pressure"]), defaults["pen_pressure"], 0, 1),
            "pen_dropout": _clamp_float(_variable_value(self.pen_dropout_var, defaults["pen_dropout"]), defaults["pen_dropout"], 0, 1),
            "text_policy": text_policy if text_policy in {"strict", "normalize"} else defaults["text_policy"],
        }

    def _current_generation_settings(self) -> dict:
        return generation_settings(
            model=str(_variable_value(self.model_var, "font")),
            writer=_variable_value(self.writer_var, "Aleatório"),
            bias=_clamp_float(_variable_value(self.bias_var, 0.75), 0.75, 0, 3),
            pen_color=str(_variable_value(self.pen_color_var, "#193787")),
            pen_width=_clamp_float(_variable_value(self.pen_width_var, 1.5), 1.5, 0.25, PEN_WIDTH_MAX),
            pen_pressure=_clamp_float(_variable_value(self.pen_pressure_var, 0.25), 0.25, 0, 1),
            pen_dropout=_clamp_float(_variable_value(self.pen_dropout_var, 0.0), 0.0, 0, 1),
            text_policy=str(_variable_value(self.text_policy_var, "strict")),
            handwriting=bool(_variable_value(self.handwriting_var, True)),
            text_handwriting=bool(_variable_value(self.text_handwriting_var, True)),
            handwriting_intensity=_clamp_float(_variable_value(self.handwriting_intensity_var, 0.65), 0.65, 0, 1),
            realistic_filters=bool(_variable_value(self.realistic_filters_var, False)),
            filter_preset=str(_variable_value(self.filter_preset_var, "medium")),
        )

    def _refresh_preset_options(self) -> None:
        previous_uids = dict(getattr(self, "style_preset_uids", {}))
        selected_style_uid = previous_uids.get(self.style_preset_var.get())
        selected_number_uid = previous_uids.get(self.number_style_var.get())
        selected_text_uid = previous_uids.get(self.text_style_var.get())
        labels = ["Configuração atual"]
        self.style_preset_uids = {"Configuração atual": None}
        for preset in list_presets():
            uid = str(preset.get("uid", ""))
            if not uid:
                continue
            label = f"{preset.get('name') or uid} · {uid}"
            labels.append(label)
            self.style_preset_uids[label] = uid
        if hasattr(self, "style_preset_box"):
            self.style_preset_box.configure(values=tuple(labels))
        if hasattr(self, "number_style_box"):
            self.number_style_box.configure(values=tuple(labels))
            self.text_style_box.configure(values=tuple(labels))
        if selected_style_uid:
            self.style_preset_var.set(self._label_for_uid(selected_style_uid))
        elif self.style_preset_var.get() not in self.style_preset_uids:
            self.style_preset_var.set("Configuração atual")
        if selected_number_uid:
            self.number_style_var.set(self._label_for_uid(selected_number_uid))
        elif self.number_style_var.get() not in self.style_preset_uids:
            self.number_style_var.set("Configuração atual")
        if selected_text_uid:
            self.text_style_var.set(self._label_for_uid(selected_text_uid))
        elif self.text_style_var.get() not in self.style_preset_uids:
            self.text_style_var.set("Configuração atual")

    def _toggle_split_presets(self) -> None:
        if not hasattr(self, "split_controls"):
            return
        if self.split_style_var.get():
            self.split_controls.grid()
        else:
            self.split_controls.grid_remove()

    def _save_current_preset(self) -> None:
        settings = self._current_generation_settings()
        name = simpledialog.askstring(
            "Preset de escrita", "Nome do preset:",
            initialvalue=suggested_preset_name(settings), parent=self,
        )
        if not name:
            return
        preset = upsert_preset({
            "uid": new_preset_uid("style"),
            "name": name.strip(),
            "created_from": "synthetic_dataset",
            "settings": settings,
        })
        self._refresh_preset_options()
        for label, uid in self.style_preset_uids.items():
            if uid == preset["uid"]:
                self.style_preset_var.set(label)
                break

    def _apply_selected_preset(self) -> None:
        uid = self.style_preset_uids.get(self.style_preset_var.get())
        if uid is None:
            return
        preset = find_preset(uid)
        settings = (preset or {}).get("settings") or {}
        if settings.get("mode") == "mixed":
            self.split_style_var.set(True)
            self._toggle_split_presets()
            self.number_style_var.set(self._label_for_uid(settings.get("number_preset_uid")))
            self.text_style_var.set(self._label_for_uid(settings.get("text_preset_uid")))
            return
        model = settings.get("model")
        if model in MODEL_PROFILES:
            self.model_var.set(model)
            self._model_changed()
        self.handwriting_var.set(bool(settings.get("handwriting", self.handwriting_var.get())))
        self.text_handwriting_var.set(bool(settings.get("text_handwriting", self.text_handwriting_var.get())))
        self.handwriting_intensity_var.set(_clamp_float(settings.get("handwriting_intensity"), self.handwriting_intensity_var.get(), 0, 1))
        self.realistic_filters_var.set(bool(settings.get("realistic_filters", self.realistic_filters_var.get())))
        preset_name = str(settings.get("filter_preset", self.filter_preset_var.get()))
        self.filter_preset_var.set(preset_name if preset_name in {"light", "medium", "strong"} else "medium")
        self.pen_color_var.set(str(settings.get("pen_color", self.pen_color_var.get())))
        self.bias_var.set(_clamp_float(settings.get("bias"), self.bias_var.get(), 0, 3))
        self.pen_width_var.set(_clamp_float(settings.get("pen_width"), self.pen_width_var.get(), 0.25, PEN_WIDTH_MAX))
        self.pen_pressure_var.set(_clamp_float(settings.get("pen_pressure"), self.pen_pressure_var.get(), 0, 1))
        self.pen_dropout_var.set(_clamp_float(settings.get("pen_dropout"), self.pen_dropout_var.get(), 0, 1))
        text_policy = settings.get("text_policy")
        self.text_policy_var.set(text_policy if text_policy in {"strict", "normalize"} else self.text_policy_var.get())
        self.writer_var.set(_valid_writer(settings.get("writer"), self.model_var.get()))
        if hasattr(self, "color_swatch"):
            self.color_swatch.configure(bg=self.pen_color_var.get())
        SyntheticDatasetFrame._capture_model_profile(self)
        self._toggle_handwriting()

    def _label_for_uid(self, uid: object) -> str:
        for label, preset_uid in self.style_preset_uids.items():
            if preset_uid == uid:
                return label
        return "Configuração atual"

    def _ensure_generation_preset(self) -> str:
        selected_uid = self.style_preset_uids.get(self.style_preset_var.get())
        if selected_uid and not self.split_style_var.get():
            return selected_uid
        if self.split_style_var.get():
            number_settings = self._settings_for_label(self.number_style_var.get())
            text_settings = self._settings_for_label(self.text_style_var.get())
            mixed_settings = {
                **self._current_generation_settings(),
                "mode": "mixed",
                "number_preset_uid": self.style_preset_uids.get(self.number_style_var.get()),
                "text_preset_uid": self.style_preset_uids.get(self.text_style_var.get()),
                "numbers": number_settings,
                "text": text_settings,
            }
            preset = upsert_preset({
                "uid": new_preset_uid("style"),
                "name": suggested_preset_name(mixed_settings),
                "created_from": "synthetic_dataset_mixed",
                "settings": mixed_settings,
            })
            self._refresh_preset_options()
            return str(preset["uid"])
        settings = self._current_generation_settings()
        preset = upsert_preset({
            "uid": new_preset_uid("style"),
            "name": suggested_preset_name(settings),
            "created_from": "synthetic_dataset_auto",
            "settings": settings,
        })
        self._refresh_preset_options()
        return str(preset["uid"])

    def _settings_for_label(self, label: str) -> dict:
        uid = self.style_preset_uids.get(label)
        if not uid:
            return self._current_generation_settings()
        preset = find_preset(uid)
        return dict((preset or {}).get("settings") or self._current_generation_settings())

    def _pen_from_settings(self, settings: dict, enabled: bool) -> NeuralPen | None:
        if not enabled or settings.get("model") == "font":
            return None
        return NeuralPen(
            model=str(settings.get("model", "sjvasquez")),
            writer=None if settings.get("writer") in (None, "", "Aleatório") else int(settings["writer"]),
            bias=float(settings.get("bias", 0.75)),
            color=str(settings.get("pen_color", "#193787")),
            width=float(settings.get("pen_width", 1.5)),
            pressure=float(settings.get("pen_pressure", 0.25)),
            dropout=float(settings.get("pen_dropout", 0.0)),
            text_policy=str(settings.get("text_policy", "normalize")),
        )

    def _apply_model_profile(self, model: str, reset: bool = False, configure_widgets: bool = True) -> None:
        if model not in MODEL_PROFILES:
            model = "font"
            self.model_var.set(model)
        profile = dict(MODEL_PROFILES.get(model, MODEL_PROFILES["font"]))
        if not reset:
            profile.update(self.model_preferences.get(model, {}))
        self.bias_var.set(_clamp_float(profile.get("bias"), MODEL_PROFILES[model]["bias"], 0, 3))
        self.pen_width_var.set(_clamp_float(profile.get("pen_width"), MODEL_PROFILES[model]["pen_width"], 0.25, PEN_WIDTH_MAX))
        self.pen_pressure_var.set(_clamp_float(profile.get("pen_pressure"), MODEL_PROFILES[model]["pen_pressure"], 0, 1))
        self.pen_dropout_var.set(_clamp_float(profile.get("pen_dropout"), MODEL_PROFILES[model]["pen_dropout"], 0, 1))
        self.text_policy_var.set(profile.get("text_policy") if profile.get("text_policy") in {"strict", "normalize"} else MODEL_PROFILES[model]["text_policy"])
        self.writer_var.set(_valid_writer(profile.get("writer"), model))
        if configure_widgets and hasattr(self, "writer_box"):
            if model == "font":
                self.writer_box.configure(values=())
            else:
                maximum = MODEL_WRITER_MAX[model]
                self.writer_box.configure(values=("Aleatório", *(str(i) for i in range(maximum + 1))))
        if hasattr(self, "color_swatch"):
            self.color_swatch.configure(bg=self.pen_color_var.get())
        SyntheticDatasetFrame._capture_model_profile(self, model)

    def _save_preferences(self) -> None:
        if self._loading_preferences:
            return
        SyntheticDatasetFrame._capture_model_profile(self)
        preferences = _read_preferences()
        model = _variable_value(self.model_var, "font")
        preferences[PREFERENCES_KEY] = {
            "count": int(_clamp_float(_variable_value(self.count_var, 10), 10, 1, 10000)),
            "seed": str(_variable_value(self.seed_var, "42")),
            "handwriting": bool(_variable_value(self.handwriting_var, True)),
            "text_handwriting": bool(_variable_value(self.text_handwriting_var, True)),
            "handwriting_intensity": _clamp_float(_variable_value(self.handwriting_intensity_var, 0.65), 0.65, 0, 1),
            "realistic_filters": bool(_variable_value(self.realistic_filters_var, False)),
            "filter_preset": str(_variable_value(self.filter_preset_var, "medium"))
            if str(_variable_value(self.filter_preset_var, "medium")) in {"light", "medium", "strong"} else "medium",
            "model": model if model in MODEL_PROFILES else "font",
            "pen_color": str(_variable_value(self.pen_color_var, "#193787")),
            "model_preferences": self.model_preferences,
        }
        try:
            _write_preferences(preferences)
        except OSError:
            pass

    def _toggle_handwriting(self) -> None:
        enabled = self.handwriting_var.get() or self.text_handwriting_var.get()
        self.handwriting_scale.configure(state="normal" if enabled and self.model_var.get() == "font" else "disabled")
        SyntheticDatasetFrame._save_preferences(self)

    def _model_changed(self, _event=None) -> None:
        model = self.model_var.get()
        previous_model = getattr(self, "current_model", None)
        if previous_model in MODEL_PROFILES and previous_model != model:
            SyntheticDatasetFrame._capture_model_profile(self, previous_model)
        self.current_model = model if model in MODEL_PROFILES else "font"
        if model == "font":
            self.neural_controls.grid_remove()
        else:
            self.neural_controls.grid()
        SyntheticDatasetFrame._apply_model_profile(self, model)
        self._toggle_handwriting()

    def _choose_pen_color(self) -> None:
        _rgb, color = colorchooser.askcolor(color=self.pen_color_var.get(), parent=self)
        if color:
            self.pen_color_var.set(color)
            self.color_swatch.configure(bg=color)
            SyntheticDatasetFrame._save_preferences(self)

    def _generate(self) -> None:
        template, mapping, output = Path(self.template_var.get()), Path(self.mapping_var.get()), Path(self.output_var.get())
        self._refresh_output_preview()
        output = Path(self.output_var.get())
        if not template.is_file() or not mapping.is_file():
            messagebox.showerror("Dataset artificial", "Selecione template e mapeamento.")
            return
        selected = self._selected_ids()
        if not selected:
            messagebox.showerror("Dataset artificial", "Selecione ao menos um campo.")
            return
        try:
            count = int(self.count_var.get())
            seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
            neural_pen = None
            numeric_neural_pen = None
            text_neural_pen = None
            if self.split_style_var.get():
                numeric_neural_pen = self._pen_from_settings(
                    self._settings_for_label(self.number_style_var.get()), self.handwriting_var.get(),
                )
                text_neural_pen = self._pen_from_settings(
                    self._settings_for_label(self.text_style_var.get()), self.text_handwriting_var.get(),
                )
            elif self.model_var.get() != "font" and (self.handwriting_var.get() or self.text_handwriting_var.get()):
                neural_pen = NeuralPen(
                    model=self.model_var.get(),
                    writer=None if self.writer_var.get() == "Aleatório" else int(self.writer_var.get()),
                    bias=float(self.bias_var.get()), color=self.pen_color_var.get(),
                    width=float(self.pen_width_var.get()), pressure=float(self.pen_pressure_var.get()),
                    dropout=float(self.pen_dropout_var.get()), text_policy=self.text_policy_var.get(),
                )
            style_preset_uid = self._ensure_generation_preset()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Dataset artificial", str(exc))
            return
        SyntheticDatasetFrame._save_preferences(self)
        self.last_generated_folder = None
        if hasattr(self, "open_viewer_button"):
            self.open_viewer_button.pack_forget()
        self._clear_generation_log()
        self._append_generation_log(f"Iniciando geração: {count} documento(s)")
        self._append_generation_log(f"Saída: {output}")
        self._append_generation_log(f"Preset: {style_preset_uid}")
        self.generate_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set(f"Gerando {count} documentos…")
        threading.Thread(
            target=self._worker,
            args=(
                template, mapping, selected, output, count, seed,
                self.handwriting_var.get(), float(self.handwriting_intensity_var.get()),
                self.text_handwriting_var.get(), neural_pen,
                numeric_neural_pen, text_neural_pen,
                self.realistic_filters_var.get(), self.filter_preset_var.get(),
                style_preset_uid,
            ),
            daemon=True,
        ).start()

    def _worker(self, template: Path, mapping: Path, selected: set[str], output: Path,
                count: int, seed: int | None, handwriting: bool,
                handwriting_intensity: float, text_handwriting: bool,
                neural_pen: NeuralPen | None = None,
                numeric_neural_pen: NeuralPen | None = None,
                text_neural_pen: NeuralPen | None = None,
                realistic_filters: bool = False,
                filter_preset: str = "medium",
                style_preset_uid: str | None = None) -> None:
        try:
            self.events.put(("log", "Carregando template e modelos..."))
            result = generate_dataset(
                template, mapping, selected, output, count, seed,
                handwriting=handwriting, handwriting_intensity=handwriting_intensity,
                text_handwriting=text_handwriting, neural_pen=neural_pen,
                numeric_neural_pen=numeric_neural_pen, text_neural_pen=text_neural_pen,
                realistic_filters=realistic_filters, filter_preset=filter_preset,
                style_preset_uid=style_preset_uid,
                progress_callback=lambda index, total, document: self.events.put((
                    "log", f"Documento {index}/{total}: {document.pdf_path.name}",
                )),
            )
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_generation_log(str(payload))
                    continue
                self.progress.stop()
                self.generate_button.configure(state="normal")
                if kind == "done":
                    generated = payload
                    self._append_generation_log(f"Concluído: {len(generated)} documento(s)")
                    self.status_var.set(f"Concluído: {len(generated)} documentos gerados em {self.output_var.get()}")
                    if generated:
                        self.last_generated_folder = generated[0].pdf_path.parent
                        self.open_viewer_button.pack(side="right", padx=(8, 0))
                        self.preview_image = pdf_to_image(str(generated[0].pdf_path), 120).image
                        self._render()
                else:
                    self._append_generation_log(f"Falha: {payload}")
                    self.status_var.set("Falha na geração.")
                    messagebox.showerror("Dataset artificial", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _open_generated_in_viewer(self) -> None:
        if self.last_generated_folder is None or self.generated_viewer is None:
            return
        self.generated_viewer.open_generated_folder(self.last_generated_folder)
        if self.generated_viewer_notebook is not None:
            self.generated_viewer_notebook.select(self.generated_viewer)

    def _clear_generation_log(self) -> None:
        if hasattr(self, "generation_log"):
            self.generation_log.delete(0, tk.END)

    def _append_generation_log(self, message: str) -> None:
        if not hasattr(self, "generation_log"):
            return
        self.generation_log.insert(tk.END, message)
        self.generation_log.see(tk.END)

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.preview_image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                text="Selecione um template e um mapeamento", fill="#a9afb8",
            )
            return
        height, width = self.preview_image.shape[:2]
        canvas_width, canvas_height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.preview_scale = min(canvas_width / width, canvas_height / height, 1.0)
        render_width, render_height = max(1, round(width * self.preview_scale)), max(1, round(height * self.preview_scale))
        offset_x, offset_y = (canvas_width - render_width) // 2, (canvas_height - render_height) // 2
        rgb = cv2.cvtColor(self.preview_image, cv2.COLOR_BGR2RGB)
        if (render_width, render_height) != (width, height):
            rgb = cv2.resize(rgb, (render_width, render_height), interpolation=cv2.INTER_AREA)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.preview_photo)
        ratio = 120 / self.mapping_dpi
        for index in self.field_list.curselection():
            x1, y1, x2, y2 = self.fields[index]["bbox"]
            self.canvas.create_rectangle(
                offset_x + x1 * ratio * self.preview_scale,
                offset_y + y1 * ratio * self.preview_scale,
                offset_x + x2 * ratio * self.preview_scale,
                offset_y + y2 * ratio * self.preview_scale,
                outline="#7c9cff", width=2,
            )
            for element in self.fields[index].get("elements", []):
                ex1, ey1, ex2, ey2 = element["bbox"]
                self.canvas.create_rectangle(
                    offset_x + ex1 * ratio * self.preview_scale,
                    offset_y + ey1 * ratio * self.preview_scale,
                    offset_x + ex2 * ratio * self.preview_scale,
                    offset_y + ey2 * ratio * self.preview_scale,
                    outline="#3ecb8a", width=2,
                )
