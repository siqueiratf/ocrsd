from __future__ import annotations

import json
import copy
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.field_rules import FIELD_TYPES
from src.pdf import pdf_to_image


UF_VALUES = (
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT",
    "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO",
    "RR", "SC", "SP", "SE", "TO",
)
DATA_TYPES = FIELD_TYPES
SEQUENCE_DATA_TYPES = {"digit_sequence", "text_sequence"}


class FieldMapperFrame(ttk.Frame):
    HANDLE_RADIUS = 6
    MOVE_HANDLE_RADIUS = 7
    MOVE_HANDLE_OFFSET = 10

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.template_image: np.ndarray | None = None
        self.template_path: Path | None = None
        self.template_id = "template"
        self.dpi = 300
        self.fields: list[dict] = []
        self.selected_index: int | None = None
        self.selected_element_index: int | None = None
        self.add_element_mode = False
        self.scale = 0.25
        self.photo: ImageTk.PhotoImage | None = None
        self._render_cache_key: tuple[int, int, int] | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_rect: int | None = None
        self.resize_handle: str | None = None
        self.resize_bbox: list[int] | None = None
        self.move_start: tuple[float, float] | None = None
        self.move_bbox: list[int] | None = None
        self.move_element_bboxes: list[list[int]] | None = None
        self.move_undo_recorded = False
        self.undo_stack: list[list[dict]] = []
        self._loading_editor = False

        self._make_variables()
        self._build_ui()
        self.bind_all("<Control-z>", self._shortcut_undo, add="+")
        self.bind_all("<Control-d>", self._shortcut_duplicate, add="+")
        self.bind_all("<Delete>", self._shortcut_delete, add="+")

    def _make_variables(self) -> None:
        self.field_id_var = tk.StringVar()
        self.label_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.data_type_var = tk.StringVar(value="text")
        self.min_length_var = tk.StringVar()
        self.max_length_var = tk.StringVar()
        self.min_value_var = tk.StringVar()
        self.max_value_var = tk.StringVar()
        self.edit_distance_var = tk.StringVar()
        self.pattern_var = tk.StringVar()
        self.allow_digits_var = tk.BooleanVar()
        self.portuguese_rules_var = tk.BooleanVar()
        self.render_as_printed_var = tk.BooleanVar()
        self.always_redact_var = tk.BooleanVar()
        self.selection_mode_var = tk.StringVar(value="single")
        self.element_value_var = tk.StringVar()
        self.element_index_var = tk.StringVar()
        self.template_status_var = tk.StringVar(value="Abra um PDF ou imagem de template.")
        self.zoom_var = tk.StringVar(value="25%")

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Button(toolbar, text="Abrir template", command=self.open_template).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Carregar JSON", command=self.load_json).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Salvar JSON", style="Accent.TButton", command=self.save_json).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Copiar JSON", command=self.copy_json).pack(side="left", padx=3)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="−", width=3, command=lambda: self.change_zoom(0.8)).pack(side="left")
        ttk.Label(toolbar, textvariable=self.zoom_var, style="Panel.TLabel", width=7, anchor="center").pack(side="left")
        ttk.Button(toolbar, text="+", width=3, command=lambda: self.change_zoom(1.25)).pack(side="left")
        ttk.Button(toolbar, text="Ajustar", command=self.fit_template).pack(side="left", padx=5)
        ttk.Label(toolbar, textvariable=self.template_status_var, style="Panel.TLabel").pack(side="left", padx=10)

        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)
        canvas_panel = ttk.Frame(paned)
        sidebar = ttk.Frame(paned, style="Panel.TFrame", width=400)
        paned.add(canvas_panel, weight=4)
        paned.add(sidebar, weight=1)

        self.canvas = tk.Canvas(canvas_panel, bg="#0e1012", highlightthickness=0)
        xbar = ttk.Scrollbar(canvas_panel, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_panel, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        canvas_panel.rowconfigure(0, weight=1)
        canvas_panel.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Motion>", self._canvas_motion)

        ttk.Label(
            sidebar, text="Campos mapeados", style="Panel.TLabel",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 5))
        self.field_list = ttk.Treeview(sidebar, columns=("id", "label", "type"), show="tree headings", height=10)
        self.field_list.heading("#0", text="Estrutura")
        self.field_list.column("#0", width=45, minwidth=35)
        for column, title, width in (("id", "ID", 55), ("label", "Campo", 180), ("type", "Tipo", 90)):
            self.field_list.heading(column, text=title)
            self.field_list.column(column, width=width, minwidth=45)
        self.field_list.pack(fill="x", padx=12)
        self.field_list.bind("<<TreeviewSelect>>", self._list_select)

        editor_canvas = tk.Canvas(sidebar, bg="#202328", highlightthickness=0)
        editor_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=editor_canvas.yview)
        editor_canvas.configure(yscrollcommand=editor_scroll.set)
        editor_scroll.pack(side="right", fill="y", pady=10)
        editor_canvas.pack(fill="both", expand=True, padx=(12, 0), pady=10)
        editor = ttk.Frame(editor_canvas, style="Panel.TFrame")
        editor_window = editor_canvas.create_window((0, 0), window=editor, anchor="nw")
        editor.bind("<Configure>", lambda _e: editor_canvas.configure(scrollregion=editor_canvas.bbox("all")))
        editor_canvas.bind("<Configure>", lambda e: editor_canvas.itemconfigure(editor_window, width=e.width))
        def scroll_editor(event: tk.Event) -> str | None:
            left, top = sidebar.winfo_rootx(), sidebar.winfo_rooty()
            if not (
                left <= event.x_root < left + sidebar.winfo_width()
                and top <= event.y_root < top + sidebar.winfo_height()
            ):
                return None
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
            editor_canvas.yview_scroll(direction, "units")
            return "break"
        editor_canvas.bind_all("<MouseWheel>", scroll_editor, add="+")
        editor_canvas.bind_all("<Button-4>", scroll_editor, add="+")
        editor_canvas.bind_all("<Button-5>", scroll_editor, add="+")
        self._build_field_editor(editor)

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="Panel.TLabel").pack(anchor="w", pady=(7, 2))
        ttk.Entry(parent, textvariable=variable).pack(fill="x")

    def _build_field_editor(self, parent: ttk.Frame) -> None:
        self._entry(parent, "ID do campo", self.field_id_var)
        self._entry(parent, "Nome/label", self.label_var)
        self._entry(parent, "Descrição", self.description_var)
        ttk.Label(parent, text="Tipo de dado", style="Panel.TLabel").pack(anchor="w", pady=(7, 2))
        type_box = ttk.Combobox(parent, textvariable=self.data_type_var, values=DATA_TYPES, state="readonly")
        type_box.pack(fill="x")
        type_box.bind("<<ComboboxSelected>>", self._type_changed)
        ttk.Label(parent, text="Seleção do checkbox", style="Panel.TLabel").pack(anchor="w", pady=(7, 2))
        ttk.Combobox(
            parent, textvariable=self.selection_mode_var,
            values=("single", "multiple"), state="readonly",
        ).pack(fill="x")
        self._entry(parent, "Valor do subelemento", self.element_value_var)
        self._entry(parent, "Índice do dígito", self.element_index_var)

        ttk.Label(parent, text="Valores permitidos (um por linha)", style="Panel.TLabel").pack(anchor="w", pady=(7, 2))
        self.allowed_values_text = tk.Text(parent, height=4, wrap="none", bg="#292d33", fg="#f2f3f5", insertbackground="#f2f3f5", relief="flat")
        self.allowed_values_text.pack(fill="x")

        limits = ttk.Frame(parent, style="Panel.TFrame")
        limits.pack(fill="x")
        for column, (label, variable) in enumerate((
            ("Tam. mín.", self.min_length_var), ("Tam. máx.", self.max_length_var),
            ("Valor mín.", self.min_value_var), ("Valor máx.", self.max_value_var),
        )):
            box = ttk.Frame(limits, style="Panel.TFrame")
            box.grid(row=column // 2, column=column % 2, sticky="ew", padx=(0, 6), pady=(7, 0))
            ttk.Label(box, text=label, style="Panel.TLabel").pack(anchor="w")
            ttk.Entry(box, textvariable=variable, width=9).pack(fill="x")
        limits.columnconfigure(0, weight=1)
        limits.columnconfigure(1, weight=1)
        self._entry(parent, "Distância máxima de correção", self.edit_distance_var)
        self._entry(parent, "Expressão regular", self.pattern_var)
        ttk.Checkbutton(parent, text="Permitir números no texto", variable=self.allow_digits_var).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(parent, text="Aplicar regras linguísticas pt-BR", variable=self.portuguese_rules_var).pack(anchor="w")
        ttk.Checkbutton(parent, text="Gerar como texto digitado", variable=self.render_as_printed_var).pack(anchor="w")
        ttk.Checkbutton(parent, text="Sempre redigir", variable=self.always_redact_var).pack(anchor="w")

        actions = ttk.Frame(parent, style="Panel.TFrame")
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Aplicar alterações", style="Accent.TButton", command=self.apply_editor).pack(side="left")
        ttk.Button(actions, text="Excluir", command=self.delete_selected).pack(side="right")
        ttk.Button(
            parent, text="Adicionar subelemento (desenhar caixa)",
            command=self.start_add_element,
        ).pack(fill="x", pady=(0, 8))
        ttk.Label(
            parent, text=(
                "Para checkbox e sequências, selecione o campo pai, clique em adicionar subelemento e desenhe sua caixa. "
                "A árvore pode ser expandida/recolhida. Arraste caixas para mover e use as alças para redimensionar. "
                "Atalhos: Ctrl+Z desfaz · Ctrl+D duplica."
            ),
            style="Panel.TLabel", wraplength=350,
        ).pack(fill="x", pady=(0, 12))

    def _event_is_in_mapper(self, event: tk.Event) -> bool:
        widget = event.widget
        while widget is not None:
            if widget is self:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _shortcut_undo(self, event: tk.Event) -> str | None:
        if not self._event_is_in_mapper(event) or isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox)
        ):
            return None
        self.undo()
        return "break"

    def _shortcut_duplicate(self, event: tk.Event) -> str | None:
        if not self._event_is_in_mapper(event) or isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox)
        ):
            return None
        self.duplicate_selected()
        return "break"

    def _shortcut_delete(self, event: tk.Event) -> str | None:
        if not self._event_is_in_mapper(event) or isinstance(
            event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Combobox)
        ):
            return None
        self.delete_selected()
        return "break"

    def _push_undo(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.fields))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.fields = self.undo_stack.pop()
        self.selected_element_index = None
        if self.selected_index is not None and self.selected_index >= len(self.fields):
            self.selected_index = len(self.fields) - 1 if self.fields else None
        self._refresh_list()
        if self.selected_index is not None:
            self.select_field(self.selected_index)
        else:
            self.render()

    def duplicate_selected(self) -> None:
        if self.selected_index is None or self.template_image is None:
            return
        self._push_undo()
        if self.selected_element_index is not None:
            parent = self.fields[self.selected_index]
            elements = parent.setdefault("elements", [])
            duplicate = copy.deepcopy(elements[self.selected_element_index])
            element_id, label, suffix = self._next_element_identity(parent)
            duplicate["element_id"] = element_id
            duplicate["label"] = label
            duplicate["bbox"] = [value + 20 for value in duplicate["bbox"]]
            if duplicate.get("index") is not None:
                duplicate["index"] = suffix
            elements.append(duplicate)
            self._refresh_list()
            self.select_element(self.selected_index, len(elements) - 1)
            return
        duplicate = copy.deepcopy(self.fields[self.selected_index])
        existing_ids = {str(field.get("field_id", "")) for field in self.fields}
        base_id = f"{duplicate.get('field_id', 'campo')}_copy"
        candidate, suffix = base_id, 2
        while candidate in existing_ids:
            candidate = f"{base_id}_{suffix}"
            suffix += 1
        duplicate["field_id"] = candidate
        duplicate["label"] = f"{duplicate.get('label', 'campo')}_copia"
        h, w = self.template_image.shape[:2]
        x1, y1, x2, y2 = duplicate["bbox"]
        width, height = x2 - x1, y2 - y1
        x1 = min(max(0, x1 + 20), max(0, w - width))
        y1 = min(max(0, y1 + 20), max(0, h - height))
        duplicate["bbox"] = [x1, y1, x1 + width, y1 + height]
        for element in duplicate.get("elements", []):
            element["bbox"] = [
                element["bbox"][0] + 20, element["bbox"][1] + 20,
                element["bbox"][2] + 20, element["bbox"][3] + 20,
            ]
        self.fields.append(duplicate)
        self._refresh_list()
        self.select_field(len(self.fields) - 1)

    @staticmethod
    def _next_element_identity(field: dict) -> tuple[str, str, int]:
        parent_id = str(field.get("field_id") or "campo")
        parent_label = str(field.get("label") or parent_id)
        pattern = re.compile(rf"^{re.escape(parent_id)}_(\d+)$")
        suffixes = []
        for element in field.get("elements", []):
            match = pattern.match(str(element.get("element_id", "")))
            if match:
                suffixes.append(int(match.group(1)))
        suffix = max(suffixes, default=-1) + 1
        return f"{parent_id}_{suffix}", f"{parent_label} {suffix}", suffix

    def open_template(self) -> None:
        path = filedialog.askopenfilename(filetypes=[
            ("Template", "*.pdf *.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*"),
        ])
        if path:
            self._load_template(Path(path))

    def _load_template(self, path: Path) -> None:
        try:
            if path.suffix.lower() == ".pdf":
                page = pdf_to_image(str(path), self.dpi)
                image = page.image
            else:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Formato de imagem inválido")
            self.template_image = image
            self._render_cache_key = None
            self.template_path = path
            self.template_id = path.stem
            self.template_status_var.set(f"{path.name} · {image.shape[1]} × {image.shape[0]} px")
            self.after_idle(self.fit_template)
        except Exception as exc:
            messagebox.showerror("Template", f"Não foi possível abrir o template: {exc}")

    def fit_template(self) -> None:
        if self.template_image is None:
            return
        self.update_idletasks()
        h, w = self.template_image.shape[:2]
        available_w = max(100, self.canvas.winfo_width() - 30)
        available_h = max(100, self.canvas.winfo_height() - 30)
        self.scale = max(0.05, min(available_w / w, available_h / h, 1.0))
        self.render()

    def change_zoom(self, factor: float) -> None:
        if self.template_image is None:
            return
        self.scale = max(0.05, min(3.0, self.scale * factor))
        self.render()

    def render(self) -> None:
        if self.template_image is None:
            self.canvas.delete("all")
            self.canvas.create_text(30, 30, anchor="nw", text="Abra um template para começar", fill="#a9afb8")
            return
        h, w = self.template_image.shape[:2]
        rw, rh = max(1, round(w * self.scale)), max(1, round(h * self.scale))
        cache_key = (id(self.template_image), rw, rh)
        if cache_key != self._render_cache_key:
            self.canvas.delete("all")
            rgb = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb, (rw, rh),
                interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_CUBIC,
            )
            self.photo = ImageTk.PhotoImage(Image.fromarray(resized))
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="template")
            self._render_cache_key = cache_key
        else:
            self.canvas.delete("overlay")
        for index, field in enumerate(self.fields):
            x1, y1, x2, y2 = (value * self.scale for value in field["bbox"])
            selected = index == self.selected_index and self.selected_element_index is None
            color = "#ff5252" if selected else ("#e69614" if field.get("always_redact") else "#7c9cff")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if selected else 2, tags="overlay")
            self._draw_move_handle(field["bbox"], color)
            if selected:
                for hx, hy in self._handles(field["bbox"]).values():
                    r = self.HANDLE_RADIUS
                    self.canvas.create_rectangle(hx-r, hy-r, hx+r, hy+r, fill="white", outline=color, tags="overlay")
            for element_index, element in enumerate(field.get("elements", [])):
                ex1, ey1, ex2, ey2 = (value * self.scale for value in element["bbox"])
                element_selected = index == self.selected_index and element_index == self.selected_element_index
                element_color = "#ff5252" if element_selected else "#3ecb8a"
                self.canvas.create_rectangle(
                    ex1, ey1, ex2, ey2, outline=element_color,
                    width=3 if element_selected else 2, tags="overlay",
                )
                self._draw_move_handle(element["bbox"], element_color)
                if element_selected:
                    for hx, hy in self._handles(element["bbox"]).values():
                        r = self.HANDLE_RADIUS
                        self.canvas.create_rectangle(hx-r, hy-r, hx+r, hy+r, fill="white", outline=element_color, tags="overlay")
        handle_padding = self.MOVE_HANDLE_OFFSET + self.MOVE_HANDLE_RADIUS + 2
        self.canvas.configure(scrollregion=(0, 0, rw + handle_padding, rh + handle_padding))
        self.zoom_var.set(f"{self.scale:.0%}")

    def _canvas_point(self, event: tk.Event) -> tuple[float, float]:
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def _handles(self, bbox: list[int]) -> dict[str, tuple[float, float]]:
        x1, y1, x2, y2 = (value * self.scale for value in bbox)
        mx, my = (x1+x2)/2, (y1+y2)/2
        return {"nw": (x1,y1), "n": (mx,y1), "ne": (x2,y1), "w": (x1,my),
                "e": (x2,my), "sw": (x1,y2), "s": (mx,y2), "se": (x2,y2)}

    def _move_handle_point(self, bbox: list[int]) -> tuple[float, float]:
        _x1, _y1, x2, y2 = (value * self.scale for value in bbox)
        return x2 + self.MOVE_HANDLE_OFFSET, y2 + self.MOVE_HANDLE_OFFSET

    def _draw_move_handle(self, bbox: list[int], color: str) -> None:
        x2, y2 = (bbox[2] * self.scale, bbox[3] * self.scale)
        hx, hy = self._move_handle_point(bbox)
        r = self.MOVE_HANDLE_RADIUS
        self.canvas.create_line(x2, y2, hx-r, hy-r, fill=color, width=2, tags="overlay")
        self.canvas.create_rectangle(
            hx-r, hy-r, hx+r, hy+r, fill=color, outline="white", width=1, tags="overlay",
        )
        self.canvas.create_line(hx-r+4, hy, hx+r-4, hy, fill="white", arrow="both", tags="overlay")
        self.canvas.create_line(hx, hy-r+4, hx, hy+r-4, fill="white", arrow="both", tags="overlay")

    def _hit_move_handle(self, x: float, y: float) -> tuple[int, int | None] | None:
        radius = self.MOVE_HANDLE_RADIUS + 3
        for index in range(len(self.fields)-1, -1, -1):
            elements = self.fields[index].get("elements", [])
            for element_index in range(len(elements)-1, -1, -1):
                hx, hy = self._move_handle_point(elements[element_index]["bbox"])
                if abs(x-hx) <= radius and abs(y-hy) <= radius:
                    return index, element_index
            hx, hy = self._move_handle_point(self.fields[index]["bbox"])
            if abs(x-hx) <= radius and abs(y-hy) <= radius:
                return index, None
        return None

    def _hit_handle(self, x: float, y: float) -> str | None:
        selected = self._selected_object()
        if selected is None:
            return None
        for name, (hx, hy) in self._handles(selected["bbox"]).items():
            if abs(x-hx) <= self.HANDLE_RADIUS * 2 and abs(y-hy) <= self.HANDLE_RADIUS * 2:
                return name
        return None

    def _hit_target(self, x: float, y: float) -> tuple[int, int | None] | None:
        ox, oy = x / self.scale, y / self.scale
        for index in range(len(self.fields)-1, -1, -1):
            elements = self.fields[index].get("elements", [])
            for element_index in range(len(elements)-1, -1, -1):
                x1, y1, x2, y2 = elements[element_index]["bbox"]
                if x1 <= ox <= x2 and y1 <= oy <= y2:
                    return index, element_index
            x1, y1, x2, y2 = self.fields[index]["bbox"]
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                return index, None
        return None

    def _selected_object(self) -> dict | None:
        if self.selected_index is None:
            return None
        field = self.fields[self.selected_index]
        if self.selected_element_index is None:
            return field
        elements = field.get("elements", [])
        return elements[self.selected_element_index] if self.selected_element_index < len(elements) else None

    def start_add_element(self) -> None:
        if self.selected_index is None or self.selected_element_index is not None:
            messagebox.showinfo("Subelemento", "Selecione primeiro o campo pai.")
            return
        field = self.fields[self.selected_index]
        if field.get("data_type") not in {"checkbox", *SEQUENCE_DATA_TYPES}:
            messagebox.showinfo(
                "Subelemento", "O campo pai deve ser checkbox, digit_sequence ou text_sequence.",
            )
            return
        self.add_element_mode = True
        self.canvas.configure(cursor="crosshair")
        self.template_status_var.set("Desenhe no documento a caixa do novo subelemento.")

    def _canvas_press(self, event: tk.Event) -> None:
        if self.template_image is None:
            return
        self.canvas.focus_set()
        x, y = self._canvas_point(event)
        move_hit = None if self.add_element_mode else self._hit_move_handle(x, y)
        if move_hit is not None:
            self._start_move(*move_hit, x, y)
            return
        handle = self._hit_handle(x, y)
        if handle and self.selected_index is not None:
            self._push_undo()
            self.resize_handle = handle
            self.resize_bbox = list(self._selected_object()["bbox"])
            return
        hit = None if self.add_element_mode else self._hit_target(x, y)
        if hit is not None:
            self._start_move(*hit, x, y)
            return
        self.drag_start = (x, y)
        self.drag_rect = self.canvas.create_rectangle(x, y, x, y, outline="#3ecb8a", width=2, tags="transient")

    def _start_move(self, field_index: int, element_index: int | None, x: float, y: float) -> None:
        if element_index is None:
            self.select_field(field_index)
        else:
            self.select_element(field_index, element_index)
        selected = self._selected_object()
        if selected is None:
            return
        self.move_start = (x, y)
        self.move_bbox = list(selected["bbox"])
        self.move_element_bboxes = (
            [list(element["bbox"]) for element in self.fields[field_index].get("elements", [])]
            if element_index is None else None
        )
        self.move_undo_recorded = False

    def _canvas_motion(self, event: tk.Event) -> None:
        if self.template_image is None or self.drag_start or self.resize_handle or self.move_start:
            return
        x, y = self._canvas_point(event)
        if self._hit_move_handle(x, y) is not None:
            self.canvas.configure(cursor="fleur")
            return
        handle = self._hit_handle(x, y)
        cursors = {
            "nw": "fleur", "se": "fleur", "ne": "fleur", "sw": "fleur",
            "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
        }
        self.canvas.configure(cursor=cursors.get(handle, "fleur" if self._hit_target(x, y) is not None else "crosshair"))

    def _canvas_drag(self, event: tk.Event) -> None:
        x, y = self._canvas_point(event)
        if self.resize_handle and self.resize_bbox is not None and self.selected_index is not None:
            ox, oy = round(x / self.scale), round(y / self.scale)
            x1, y1, x2, y2 = self.resize_bbox
            handle = self.resize_handle
            if "n" in handle: y1 = oy
            if "s" in handle: y2 = oy
            if "w" in handle: x1 = ox
            if "e" in handle: x2 = ox
            selected = self._selected_object()
            if selected is not None:
                selected["bbox"] = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
            self.render()
        elif self.move_start and self.move_bbox is not None and self.selected_index is not None:
            dx = round((x - self.move_start[0]) / self.scale)
            dy = round((y - self.move_start[1]) / self.scale)
            if dx == 0 and dy == 0:
                return
            if not self.move_undo_recorded:
                self._push_undo()
                self.move_undo_recorded = True
            x1, y1, x2, y2 = self.move_bbox
            width, height = x2 - x1, y2 - y1
            image_h, image_w = self.template_image.shape[:2]
            moved_x1 = min(max(0, x1 + dx), max(0, image_w - width))
            moved_y1 = min(max(0, y1 + dy), max(0, image_h - height))
            selected = self._selected_object()
            if selected is not None:
                selected["bbox"] = [moved_x1, moved_y1, moved_x1 + width, moved_y1 + height]
                if self.selected_element_index is None and self.move_element_bboxes is not None:
                    actual_dx, actual_dy = moved_x1 - x1, moved_y1 - y1
                    for element, original in zip(
                        self.fields[self.selected_index].get("elements", []), self.move_element_bboxes,
                    ):
                        element["bbox"] = [
                            original[0] + actual_dx, original[1] + actual_dy,
                            original[2] + actual_dx, original[3] + actual_dy,
                        ]
            self.render()
        elif self.drag_start and self.drag_rect:
            self.canvas.coords(self.drag_rect, self.drag_start[0], self.drag_start[1], x, y)

    def _canvas_release(self, event: tk.Event) -> None:
        if self.resize_handle:
            self._canvas_drag(event)
            self.resize_handle = None
            self.resize_bbox = None
            self._refresh_list()
            return
        if self.move_start:
            self._canvas_drag(event)
            self.move_start = None
            self.move_bbox = None
            self.move_element_bboxes = None
            self.move_undo_recorded = False
            self._refresh_list()
            return
        if not self.drag_start:
            return
        x, y = self._canvas_point(event)
        sx, sy = self.drag_start
        self.drag_start = None
        if self.drag_rect is not None:
            self.canvas.delete(self.drag_rect)
        self.drag_rect = None
        if abs(x-sx) < 5 or abs(y-sy) < 5:
            self.render()
            return
        h, w = self.template_image.shape[:2]
        bbox = [
            max(0, round(min(sx,x)/self.scale)), max(0, round(min(sy,y)/self.scale)),
            min(w, round(max(sx,x)/self.scale)), min(h, round(max(sy,y)/self.scale)),
        ]
        self._push_undo()
        if self.add_element_mode and self.selected_index is not None:
            field = self.fields[self.selected_index]
            elements = field.setdefault("elements", [])
            element_index = len(elements)
            element_id, label, suffix = self._next_element_identity(field)
            element = {
                "element_id": element_id, "label": label, "bbox": bbox,
            }
            if field.get("data_type") in SEQUENCE_DATA_TYPES:
                element["index"] = suffix
            elements.append(element)
            self.add_element_mode = False
            self._refresh_list()
            self.select_element(self.selected_index, element_index)
            return
        field = {"field_id": str(len(self.fields)+1), "label": "novo_campo", "description": "",
                 "bbox": bbox, "data_type": "text", "validation": {}}
        self.fields.append(field)
        self._refresh_list()
        self.select_field(len(self.fields)-1)

    def _refresh_list(self) -> None:
        for item in self.field_list.get_children():
            self.field_list.delete(item)
        for index, field in enumerate(self.fields):
            field_iid = f"field:{index}"
            self.field_list.insert("", "end", iid=field_iid, text="▸", open=True, values=(
                field.get("field_id", ""), field.get("label", ""), field.get("data_type", "text"),
            ))
            for element_index, element in enumerate(field.get("elements", [])):
                self.field_list.insert(field_iid, "end", iid=f"element:{index}:{element_index}", text="•", values=(
                    element.get("element_id", ""), element.get("label", ""), "subelemento",
                ))

    def _list_select(self, _event: tk.Event) -> None:
        if self._loading_editor:
            return
        selection = self.field_list.selection()
        if selection:
            parts = selection[0].split(":")
            if parts[0] == "field":
                field_index = int(parts[1])
                if self.selected_index == field_index and self.selected_element_index is None:
                    return
                self.select_field(field_index)
            else:
                field_index, element_index = int(parts[1]), int(parts[2])
                if (
                    self.selected_index == field_index
                    and self.selected_element_index == element_index
                ):
                    return
                self.select_element(field_index, element_index)

    def select_field(self, index: int) -> None:
        self.selected_index = index
        self.selected_element_index = None
        self._loading_editor = True
        field = self.fields[index]
        validation = field.get("validation") or {}
        self.field_id_var.set(field.get("field_id", ""))
        self.label_var.set(field.get("label", ""))
        self.description_var.set(field.get("description", ""))
        self.data_type_var.set(field.get("data_type", "text"))
        self.selection_mode_var.set(field.get("selection_mode", "single"))
        self.element_value_var.set("")
        self.element_index_var.set("")
        self.allowed_values_text.delete("1.0", "end")
        self.allowed_values_text.insert("1.0", "\n".join(validation.get("allowed_values", [])))
        for variable, key in ((self.min_length_var,"min_length"), (self.max_length_var,"max_length"),
                              (self.min_value_var,"min_value"), (self.max_value_var,"max_value"),
                              (self.edit_distance_var,"max_edit_distance"), (self.pattern_var,"pattern")):
            variable.set("" if validation.get(key) is None else str(validation[key]))
        self.allow_digits_var.set(bool(validation.get("allow_digits", False)))
        self.portuguese_rules_var.set(bool(validation.get("portuguese_rules", False)))
        self.render_as_printed_var.set(bool(field.get("render_as_printed", False)))
        self.always_redact_var.set(bool(field.get("always_redact", False)))
        self._loading_editor = False
        iid = f"field:{index}"
        self.field_list.selection_set(iid)
        self.field_list.see(iid)
        self.render()

    def select_element(self, field_index: int, element_index: int) -> None:
        self.selected_index = field_index
        self.selected_element_index = element_index
        self._loading_editor = True
        field = self.fields[field_index]
        element = field.get("elements", [])[element_index]
        self.field_id_var.set(element.get("element_id", ""))
        self.label_var.set(element.get("label", ""))
        self.description_var.set("")
        self.data_type_var.set(field.get("data_type", "text"))
        self.selection_mode_var.set(field.get("selection_mode", "single"))
        self.element_value_var.set(element.get("value", ""))
        self.element_index_var.set("" if element.get("index") is None else str(element["index"]))
        self.allowed_values_text.delete("1.0", "end")
        for variable in (
            self.min_length_var, self.max_length_var, self.min_value_var,
            self.max_value_var, self.edit_distance_var, self.pattern_var,
        ):
            variable.set("")
        self.allow_digits_var.set(False)
        self.portuguese_rules_var.set(False)
        self.render_as_printed_var.set(False)
        self.always_redact_var.set(False)
        self._loading_editor = False
        iid = f"element:{field_index}:{element_index}"
        self.field_list.selection_set(iid)
        self.field_list.see(iid)
        self.render()

    def _type_changed(self, _event: tk.Event | None = None) -> None:
        if self._loading_editor:
            return
        data_type = self.data_type_var.get()
        if data_type == "uf":
            self.allowed_values_text.delete("1.0", "end")
            self.allowed_values_text.insert("1.0", "\n".join(UF_VALUES))
            self.edit_distance_var.set("1")
        elif data_type == "person_name":
            self.allow_digits_var.set(False)
            self.portuguese_rules_var.set(True)
            self.pattern_var.set("^[A-Za-zÀ-ÖØ-öø-ÿ' -]+$")
        elif data_type == "checkbox":
            self.selection_mode_var.set("single")
        elif data_type == "text_sequence":
            self.pattern_var.set("^[A-Z]+$")

    @staticmethod
    def _optional_number(value: str, integer: bool = False):
        if not value.strip():
            return None
        return int(value) if integer else float(value)

    def apply_editor(self) -> bool:
        if self.selected_index is None:
            messagebox.showinfo("Campo", "Selecione ou desenhe um campo primeiro.")
            return False
        try:
            field_id, label = self.field_id_var.get().strip(), self.label_var.get().strip()
            if not field_id or not label:
                raise ValueError("ID e nome do campo são obrigatórios")
            if self.selected_element_index is not None:
                elements = self.fields[self.selected_index].get("elements", [])
                if any(
                    index != self.selected_element_index and element.get("element_id") == field_id
                    for index, element in enumerate(elements)
                ):
                    raise ValueError(f"Já existe um subelemento com o ID '{field_id}'")
                self._push_undo()
                element = elements[self.selected_element_index]
                element.update(element_id=field_id, label=label)
                value = self.element_value_var.get().strip()
                if value:
                    element["value"] = value
                else:
                    element.pop("value", None)
                index_value = self.element_index_var.get().strip()
                if index_value:
                    element["index"] = int(index_value)
                else:
                    element.pop("index", None)
                self._refresh_list()
                self.select_element(self.selected_index, self.selected_element_index)
                return True
            if any(
                index != self.selected_index and field.get("field_id") == field_id
                for index, field in enumerate(self.fields)
            ):
                raise ValueError(f"Já existe um campo com o ID '{field_id}'")
            allowed = []
            for value in self.allowed_values_text.get("1.0", "end").replace(",", "\n").replace(";", "\n").splitlines():
                value = value.strip()
                if value and value not in allowed:
                    allowed.append(value)
            validation = {
                "allowed_values": allowed,
                "min_length": self._optional_number(self.min_length_var.get(), True),
                "max_length": self._optional_number(self.max_length_var.get(), True),
                "min_value": self._optional_number(self.min_value_var.get()),
                "max_value": self._optional_number(self.max_value_var.get()),
                "max_edit_distance": self._optional_number(self.edit_distance_var.get(), True),
                "pattern": self.pattern_var.get().strip() or None,
                "allow_digits": self.allow_digits_var.get() if self.data_type_var.get() == "person_name" else None,
                "portuguese_rules": self.portuguese_rules_var.get() or None,
                "language": "pt-BR" if self.portuguese_rules_var.get() else None,
            }
            validation = {key: value for key, value in validation.items() if value is not None and value != []}
            self._push_undo()
            field = self.fields[self.selected_index]
            field.update(field_id=field_id, label=label, description=self.description_var.get().strip(),
                         data_type=self.data_type_var.get(), validation=validation,
                         selection_mode=self.selection_mode_var.get())
            if self.always_redact_var.get():
                field["always_redact"] = True
            else:
                field.pop("always_redact", None)
            if self.render_as_printed_var.get():
                field["render_as_printed"] = True
            else:
                field.pop("render_as_printed", None)
            self._refresh_list()
            self.select_field(self.selected_index)
            return True
        except Exception as exc:
            messagebox.showerror("Campo", str(exc))
            return False

    def delete_selected(self) -> None:
        if self.selected_index is None:
            return
        self._push_undo()
        if self.selected_element_index is not None:
            del self.fields[self.selected_index].setdefault("elements", [])[self.selected_element_index]
            field_index = self.selected_index
            self.selected_element_index = None
            self._refresh_list()
            self.select_field(field_index)
            return
        del self.fields[self.selected_index]
        self.selected_index = None
        self._refresh_list()
        self.render()

    def build_field_map(self) -> dict:
        if self.template_image is None:
            raise ValueError("Abra o template antes de exportar")
        h, w = self.template_image.shape[:2]
        clean_fields = []
        for field in self.fields:
            item = {"field_id": field["field_id"], "label": field["label"],
                    "description": field.get("description", ""), "bbox": field["bbox"]}
            if field.get("always_redact"): item["always_redact"] = True
            if field.get("render_as_printed"): item["render_as_printed"] = True
            if field.get("data_type", "text") != "text": item["data_type"] = field["data_type"]
            if field.get("validation"): item["validation"] = field["validation"]
            if field.get("data_type") == "checkbox":
                item["selection_mode"] = field.get("selection_mode", "single")
            if field.get("elements"):
                item["elements"] = copy.deepcopy(field["elements"])
            clean_fields.append(item)
        return {"template_id": self.template_id, "dpi": self.dpi, "page_size_px": [w, h],
                "source_template": str(self.template_path) if self.template_path else "", "fields": clean_fields}

    def save_json(self) -> None:
        if self.selected_index is not None and not self.apply_editor():
            return
        try:
            data = self.build_field_map()
        except Exception as exc:
            messagebox.showerror("Mapa de campos", str(exc)); return
        mappings_dir = (
            self.template_path.parent / "mappings"
            if self.template_path and self.template_path.name == "template.pdf"
            else None
        )
        path = filedialog.asksaveasfilename(
            initialdir=str(mappings_dir) if mappings_dir and mappings_dir.is_dir() else None,
            defaultextension=".json", initialfile=f"{self.template_id}_fields.json",
            filetypes=[("JSON", "*.json")],
        )
        if path:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            self.template_status_var.set(f"Mapa salvo em {path}")

    def copy_json(self) -> None:
        if self.selected_index is not None and not self.apply_editor():
            return
        try:
            value = json.dumps(self.build_field_map(), ensure_ascii=False, indent=2)
            self.clipboard_clear(); self.clipboard_append(value)
            self.template_status_var.set("JSON copiado para a área de transferência")
        except Exception as exc:
            messagebox.showerror("Mapa de campos", str(exc))

    def load_json(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data.get("fields"), list):
                raise ValueError("JSON sem uma lista 'fields'")
            self.fields = data["fields"]
            self.undo_stack.clear()
            map_template_id = data.get("template_id", Path(path).stem)
            self.template_id = map_template_id
            self.dpi = int(data.get("dpi", 300))
            source = data.get("source_template")
            candidates = [Path(source)] if source else []
            if source:
                candidates.append(Path(path).parent / source)
            template = next((candidate for candidate in candidates if candidate.is_file()), None)
            if template:
                self._load_template(template)
                self.template_id = map_template_id
            elif self.template_image is None:
                messagebox.showinfo("Template", "Mapa carregado. Agora abra o PDF ou imagem correspondente.")
            self.selected_index = None
            self._refresh_list(); self.render()
        except Exception as exc:
            messagebox.showerror("Mapa de campos", f"Não foi possível carregar o JSON: {exc}")
