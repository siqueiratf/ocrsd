from __future__ import annotations

import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import fitz
from PIL import Image, ImageTk

from src.document_types import discover_document_types
from src.writing_presets import find_preset, preset_summary, uid_from_pdf_name


BG = "#15171a"
PANEL = "#202328"
PANEL_2 = "#292d33"
TEXT = "#f2f3f5"
MUTED = "#a9afb8"
ACCENT = "#7c9cff"


def _folder_mtime(folder: Path) -> float:
    items = [folder]
    try:
        items.extend(item for item in folder.iterdir() if item.is_file())
    except OSError:
        pass
    return max((item.stat().st_mtime for item in items if item.exists()), default=0.0)


def find_recent_dataset_folders(limit: int = 3, root: Path | None = None) -> list[Path]:
    root = root or Path.cwd()
    search_roots = [root / "documents"]
    for document_type in discover_document_types().values():
        search_roots.append(document_type.dataset)
    manifests: dict[Path, Path] = {}
    for base in search_roots:
        if not base.is_dir():
            continue
        for manifest in base.rglob("dataset.json"):
            manifests[manifest.parent.resolve()] = manifest
    folders = sorted(manifests, key=lambda folder: _folder_mtime(folder), reverse=True)
    return folders[:limit]


def collect_pdfs(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)


def pdf_display_name(pdf: Path, root: Path) -> str:
    try:
        relative = pdf.relative_to(root)
    except ValueError:
        return pdf.name
    return str(relative) if relative.parent != Path(".") else pdf.name


def manifest_folder_for_pdf(pdf: Path, root: Path) -> Path:
    for folder in (pdf.parent, *pdf.parents):
        if (folder / "dataset.json").is_file():
            return folder
        if folder == root:
            break
    return pdf.parent


class GeneratedPdfViewerFrame(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.folders: list[Path] = []
        self.pdfs: list[Path] = []
        self.current_pdf: Path | None = None
        self.current_folder: Path | None = None
        self.document: fitz.Document | None = None
        self.page_index = 0
        self.zoom = 1.0
        self.fit_mode = "page"
        self.photo: ImageTk.PhotoImage | None = None
        self.status_var = tk.StringVar(value="Carregue os últimos datasets gerados.")
        self.folder_var = tk.StringVar(value="")
        self.page_var = tk.StringVar(value="0 / 0")
        self.zoom_var = tk.StringVar(value="100%")
        self.meta_var = tk.StringVar(value="")
        self.dataset_meta_text = ""
        self._build_ui()

    def destroy(self) -> None:
        self._close_document()
        super().destroy()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Panel.TFrame", padding=10)
        header.pack(fill="x", pady=(10, 8))
        ttk.Label(header, text="Visualizador de PDFs", style="Panel.TLabel", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var, style="Panel.TLabel", anchor="center").pack(side="left", fill="x", expand=True, padx=12)
        ttk.Button(header, text="Carregar últimos gerados", style="Accent.TButton", command=self.load_recent).pack(side="right")
        ttk.Button(header, text="Abrir pasta…", command=self.open_folder).pack(side="right", padx=(0, 6))

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, style="Panel.TFrame", width=340)
        viewer = ttk.Frame(body)
        body.add(sidebar, weight=1)
        body.add(viewer, weight=4)

        ttk.Label(sidebar, text="Pastas", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        self.folder_list = tk.Listbox(
            sidebar, exportselection=False, height=6, bg=PANEL_2, fg=TEXT,
            selectbackground="#34446f", relief="flat",
        )
        self.folder_list.pack(fill="x", padx=12, pady=(0, 8))
        self.folder_list.bind("<<ListboxSelect>>", lambda _event: self._folder_selected())

        folder_actions = ttk.Frame(sidebar, style="Panel.TFrame")
        folder_actions.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(folder_actions, text="Atualizar", command=self.reload_current_folder).pack(side="left")
        ttk.Button(folder_actions, text="Excluir pasta", command=self.delete_current_folder).pack(side="right")

        ttk.Label(sidebar, text="PDFs", style="Panel.TLabel", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=12, pady=(0, 4))
        self.pdf_list = tk.Listbox(
            sidebar, exportselection=False, height=15, bg=PANEL_2, fg=TEXT,
            selectbackground="#34446f", relief="flat",
        )
        self.pdf_list.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.pdf_list.bind("<<ListboxSelect>>", lambda _event: self._pdf_selected())
        self.pdf_list.bind("<Double-Button-1>", lambda _event: self.fit_page())

        pdf_actions = ttk.Frame(sidebar, style="Panel.TFrame")
        pdf_actions.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(pdf_actions, text="Renomear", command=self.rename_current_pdf).pack(side="left")
        ttk.Button(pdf_actions, text="Excluir", command=self.delete_current_pdf).pack(side="right")

        ttk.Label(sidebar, textvariable=self.folder_var, style="Panel.TLabel", wraplength=300).pack(fill="x", padx=12, pady=(0, 8))
        ttk.Label(sidebar, textvariable=self.meta_var, style="Panel.TLabel", foreground=MUTED, wraplength=300).pack(fill="x", padx=12, pady=(0, 12))

        toolbar = ttk.Frame(viewer, style="Panel.TFrame", padding=10)
        toolbar.pack(fill="x", pady=(0, 8))
        for label, command in (
            ("‹", self.previous_page),
            ("›", self.next_page),
            ("−", lambda: self.change_zoom(0.85)),
            ("+", lambda: self.change_zoom(1.18)),
        ):
            ttk.Button(toolbar, text=label, width=3, command=command).pack(side="left", padx=2)
        ttk.Label(toolbar, textvariable=self.page_var, style="Panel.TLabel", width=9, anchor="center").pack(side="left", padx=8)
        ttk.Label(toolbar, textvariable=self.zoom_var, style="Panel.TLabel", width=7, anchor="center").pack(side="left", padx=6)
        ttk.Button(toolbar, text="Ajustar página", command=self.fit_page).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Ajustar largura", command=self.fit_width).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Recarregar PDF", command=self.reload_current_pdf).pack(side="right")

        canvas_frame = ttk.Frame(viewer)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#0e1012", highlightthickness=0)
        xbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        ybar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Control-Button-4>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Control-Button-5>", self._on_ctrl_mousewheel)

    def load_recent(self) -> None:
        self._set_folders(find_recent_dataset_folders(3))
        if not self.folders:
            self.status_var.set("Nenhum dataset gerado encontrado.")

    def open_folder(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.open_generated_folder(Path(selected))

    def open_generated_folder(self, folder: Path) -> None:
        folder = folder.resolve()
        folders = [folder, *(item for item in self.folders if item != folder)]
        self._set_folders(folders[:3])

    def _set_folders(self, folders: list[Path]) -> None:
        self.folders = folders
        self.folder_list.delete(0, tk.END)
        for folder in folders:
            self.folder_list.insert(tk.END, folder.name)
        if folders:
            self.folder_list.selection_set(0)
            self._load_folder(folders[0])

    def _folder_selected(self) -> None:
        selection = self.folder_list.curselection()
        if selection:
            self._load_folder(self.folders[selection[0]])

    def _load_folder(self, folder: Path) -> None:
        self.current_folder = folder
        self.folder_var.set(str(folder))
        self.pdfs = collect_pdfs(folder)
        self.pdf_list.delete(0, tk.END)
        for pdf in self.pdfs:
            self.pdf_list.insert(tk.END, pdf_display_name(pdf, folder))
        self._load_manifest(folder)
        if self.pdfs:
            self.pdf_list.selection_set(0)
            self.open_pdf(self.pdfs[0])
            self.status_var.set(f"{len(self.pdfs)} PDF(s) em {folder.name}, incluindo subpastas")
        else:
            self._close_document()
            self.current_pdf = None
            self.status_var.set(f"Nenhum PDF encontrado em {folder.name}")
            self._render()

    def _load_manifest(self, folder: Path) -> None:
        manifest = folder / "dataset.json"
        if not manifest.is_file():
            self.dataset_meta_text = "Sem dataset.json nesta pasta."
            self.meta_var.set(self.dataset_meta_text)
            return
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.dataset_meta_text = f"dataset.json inválido: {exc}"
            self.meta_var.set(self.dataset_meta_text)
            return
        count = data.get("count", "?")
        seed = data.get("seed", "")
        template = Path(str(data.get("template", ""))).name if data.get("template") else ""
        self.dataset_meta_text = f"Documentos: {count}\nSeed: {seed}\nTemplate: {template}"
        self.meta_var.set(self.dataset_meta_text)

    def _pdf_selected(self) -> None:
        selection = self.pdf_list.curselection()
        if selection:
            self.open_pdf(self.pdfs[selection[0]])

    def open_pdf(self, path: Path) -> None:
        try:
            self._close_document()
            self.document = fitz.open(path)
            self.current_pdf = path
            self.page_index = 0
            self.fit_mode = "page"
            self._render()
            self.status_var.set(path.name)
            self._show_pdf_preset(path)
        except Exception as exc:
            messagebox.showerror("Visualizador de PDFs", str(exc))

    def _show_pdf_preset(self, path: Path) -> None:
        uid = uid_from_pdf_name(path.name)
        if uid is None:
            self.meta_var.set(f"{self.dataset_meta_text}\n\nPreset de escrita: não informado no nome do PDF")
            return
        self.meta_var.set(f"{self.dataset_meta_text}\n\n{preset_summary(find_preset(uid))}")

    def reload_current_folder(self) -> None:
        selection = self.folder_list.curselection()
        if selection:
            self._load_folder(self.folders[selection[0]])

    def reload_current_pdf(self) -> None:
        if self.current_pdf is not None:
            current = self.current_pdf
            self.open_pdf(current)

    def rename_current_pdf(self) -> None:
        if self.current_pdf is None:
            return
        new_name = simpledialog.askstring(
            "Renomear PDF", "Novo nome do PDF:", initialvalue=self.current_pdf.stem, parent=self,
        )
        if not new_name:
            return
        cleaned = "".join(character for character in new_name if character not in '\\/:*?"<>|').strip()
        if not cleaned:
            messagebox.showerror("Renomear PDF", "Informe um nome válido.")
            return
        target = self.current_pdf.with_name(f"{cleaned}.pdf")
        if target.exists():
            messagebox.showerror("Renomear PDF", "Já existe um PDF com esse nome.")
            return
        source = self.current_pdf
        json_source = source.with_suffix(".json")
        json_target = target.with_suffix(".json")
        manifest_folder = manifest_folder_for_pdf(source, self.current_folder or source.parent)
        try:
            source.rename(target)
            if json_source.exists():
                json_source.rename(json_target)
            self._replace_manifest_document(manifest_folder, source.name, target.name, json_target.name)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Renomear PDF", str(exc))
            return
        self.current_pdf = target
        self.reload_current_folder()
        self._select_pdf(target)

    def delete_current_pdf(self) -> None:
        if self.current_pdf is None:
            return
        if not messagebox.askyesno("Excluir PDF", f"Excluir {self.current_pdf.name} e o JSON correspondente?"):
            return
        root_folder = self.current_folder or self.current_pdf.parent
        manifest_folder = manifest_folder_for_pdf(self.current_pdf, root_folder)
        try:
            self.current_pdf.unlink(missing_ok=True)
            self.current_pdf.with_suffix(".json").unlink(missing_ok=True)
            self._remove_manifest_document(manifest_folder, self.current_pdf.name)
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Excluir PDF", str(exc))
            return
        self.current_pdf = None
        self._load_folder(root_folder)

    def delete_current_folder(self) -> None:
        selection = self.folder_list.curselection()
        if not selection:
            return
        folder = self.folders[selection[0]]
        if not messagebox.askyesno("Excluir pasta", f"Excluir a pasta inteira?\n{folder}"):
            return
        self._close_document()
        try:
            shutil.rmtree(folder)
        except OSError as exc:
            messagebox.showerror("Excluir pasta", str(exc))
            return
        self._set_folders([item for item in self.folders if item != folder])
        self.status_var.set("Pasta excluída.")

    def previous_page(self) -> None:
        if self.document is not None and self.page_index > 0:
            self.page_index -= 1
            self._render()

    def next_page(self) -> None:
        if self.document is not None and self.page_index < len(self.document) - 1:
            self.page_index += 1
            self._render()

    def change_zoom(self, factor: float) -> None:
        self.fit_mode = "manual"
        self.zoom = min(5.0, max(0.15, self.zoom * factor))
        self._render()

    def fit_page(self) -> None:
        self.fit_mode = "page"
        self._render()

    def fit_width(self) -> None:
        self.fit_mode = "width"
        self._render()

    def _on_ctrl_mousewheel(self, event: tk.Event) -> str:
        direction = 1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else -1
        self.change_zoom(1.12 if direction > 0 else 0.89)
        return "break"

    def _on_mousewheel(self, event: tk.Event) -> str:
        if getattr(event, "state", 0) & 0x4:
            return self._on_ctrl_mousewheel(event)
        direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", 0) == 4 else 1
        if getattr(event, "state", 0) & 0x1:
            self.canvas.xview_scroll(direction, "units")
        else:
            self.canvas.yview_scroll(direction, "units")
        return "break"

    def _select_pdf(self, path: Path) -> None:
        for index, pdf in enumerate(self.pdfs):
            if pdf == path:
                self.pdf_list.selection_clear(0, tk.END)
                self.pdf_list.selection_set(index)
                self.pdf_list.see(index)
                self.open_pdf(path)
                return

    def _replace_manifest_document(self, folder: Path, old_pdf: str, new_pdf: str, new_json: str) -> None:
        manifest = folder / "dataset.json"
        if not manifest.is_file():
            return
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for document in data.get("documents", []):
            if document.get("pdf") == old_pdf:
                document["pdf"] = new_pdf
                document["ground_truth"] = new_json
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _remove_manifest_document(self, folder: Path, pdf_name: str) -> None:
        manifest = folder / "dataset.json"
        if not manifest.is_file():
            return
        data = json.loads(manifest.read_text(encoding="utf-8"))
        documents = [document for document in data.get("documents", []) if document.get("pdf") != pdf_name]
        data["documents"] = documents
        data["count"] = len(documents)
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _close_document(self) -> None:
        if self.document is not None:
            self.document.close()
        self.document = None
        self.photo = None

    def _render(self) -> None:
        self.canvas.delete("all")
        if self.document is None or len(self.document) == 0:
            self.page_var.set("0 / 0")
            self.zoom_var.set("100%")
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2,
                text="Nenhum PDF selecionado",
                fill=MUTED,
            )
            return
        page = self.document[self.page_index]
        rect = page.rect
        canvas_width = max(1, self.canvas.winfo_width() - 32)
        canvas_height = max(1, self.canvas.winfo_height() - 32)
        if self.fit_mode == "page":
            self.zoom = min(canvas_width / rect.width, canvas_height / rect.height)
        elif self.fit_mode == "width":
            self.zoom = canvas_width / rect.width
        self.zoom = min(5.0, max(0.15, self.zoom))
        matrix = fitz.Matrix(self.zoom, self.zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        self.photo = ImageTk.PhotoImage(image)
        x = max(16, (self.canvas.winfo_width() - pixmap.width) // 2)
        y = max(16, (self.canvas.winfo_height() - pixmap.height) // 2)
        self.canvas.create_rectangle(x - 1, y - 1, x + pixmap.width + 1, y + pixmap.height + 1, fill="#f7f7f7", outline="#383d45")
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, max(self.canvas.winfo_width(), x + pixmap.width + 16), max(self.canvas.winfo_height(), y + pixmap.height + 16)))
        self.page_var.set(f"{self.page_index + 1} / {len(self.document)}")
        self.zoom_var.set(f"{self.zoom:.0%}")
