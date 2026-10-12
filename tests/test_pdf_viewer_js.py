from pathlib import Path
import shutil
import subprocess
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/web/static/pdf_viewer.js"
HARNESS = """
const assert = require('assert');
let observer = null;
global.ResizeObserver = class { constructor(callback) { observer = callback } observe() {} disconnect() {} };
global.getComputedStyle = () => ({paddingLeft: '20px', paddingRight: '20px', paddingTop: '20px', paddingBottom: '20px'});
global.document = {fullscreenElement: null, getElementById: () => null};
const stage = {
  clientWidth: 900, clientHeight: 650, clientLeft: 0, clientTop: 0, scrollLeft: 0, scrollTop: 0, scrollWidth: 900, scrollHeight: 650,
  addEventListener() {}, scrollTo() {}, classList: {add() {}, remove() {}},
  getBoundingClientRect() { return {left: 0, top: 0, width: this.clientWidth, height: this.clientHeight} },
};
const {createViewer, fitScale, clampZoom, ZOOM_STEP} = require(process.argv[1]);
"""


class PDFViewerJavaScriptTests(unittest.TestCase):
    def run_node(self, script: str):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js indisponível")
        result = subprocess.run([node, "-e", HARNESS + script, str(SOURCE)], capture_output=True, text=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fit_scale_uses_intrinsic_size(self):
        self.run_node("""
const first = fitScale(900, 650, 1800, 2400, 'page', 1, 40);
assert.strictEqual(first, fitScale(900, 650, 1800, 2400, 'page', 1, 40));
assert.ok(first > 0 && first < 1);
assert.strictEqual(fitScale(900, 650, 1800, 2400, 'width', 1, 40), 860 / 1800);
assert.strictEqual(clampZoom(9), 3);
assert.strictEqual(clampZoom(0.01), 0.1);
""")

    def test_fit_page_does_not_shrink_when_stage_follows_content(self):
        self.run_node("""
const scales = [];
const viewer = createViewer({stage, natural: () => [1800, 2400], view: {mode: 'page', scale: 1},
  render: scale => { scales.push(scale); stage.clientHeight = Math.round(2400 * scale) + 30 }});
for (let index = 0; index < 200; index++) observer();
assert.strictEqual(scales.length, 1);
assert.strictEqual(viewer.scale(), scales[0]);
stage.clientWidth = 1400; stage.clientHeight = 900;
observer(); observer(); observer();
assert.strictEqual(scales.length, 2);
assert.ok(scales[1] > scales[0]);
""")

    def test_manual_zoom_is_kept_after_resize(self):
        self.run_node("""
let rendered = 0;
const viewer = createViewer({stage, natural: () => [1800, 2400], view: {mode: 'page', scale: 1}, render: scale => { rendered = scale }});
const fitted = viewer.scale();
viewer.actions['zoom-in']();
assert.ok(Math.abs(viewer.scale() - fitted * ZOOM_STEP) < 1e-9);
assert.strictEqual(viewer.view().mode, 'manual');
stage.clientWidth = 500; stage.clientHeight = 400;
observer();
assert.ok(Math.abs(rendered - fitted * ZOOM_STEP) < 1e-9);
viewer.actions['fit-width']();
assert.strictEqual(viewer.scale(), (500 - 40) / 1800);
""")
