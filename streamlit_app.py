
import runpy, sys, io, traceback
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Visuals", layout="wide")
st.title("EDA Visuals (from EDA.ipynb) — Full View")

# ---------- INSIGHTS FROM NOTEBOOK (Markdown) ----------
try:
    import nbformat
    nb_path = Path("EDA.ipynb")
    if nb_path.exists():
        nb = nbformat.read(str(nb_path), as_version=4)
        md_cells = [cell for cell in nb.cells if cell.cell_type == "markdown"]
        if md_cells:
            st.subheader("Insights from Notebook")
            for i, cell in enumerate(md_cells, 1):
                st.markdown(f"### Insight Section {i}")
                st.markdown(cell.source)
except Exception as e:
    st.warning(f"Could not load insights: {e}")

# ---------- RUN EDA SCRIPT ----------
out_dir = Path("plots_streamlit")
out_dir.mkdir(exist_ok=True)

buf = io.StringIO()
err = io.StringIO()
sys_stdout = sys.stdout
sys_stderr = sys.stderr

# Capture Plotly shows
captured_plotly = []
try:
    import plotly.io as pio
    original_plotly_show = pio.show

    def patched_show(fig, *args, **kwargs):
        try:
            captured_plotly.append(fig)
        except:
            pass

    pio.show = patched_show
except:
    pio = None

try:
    import altair as alt
    has_altair = True
except:
    alt = None
    has_altair = False

# Patch plt.show
_original_plt_show = plt.show
def patched_plt_show(*args, **kwargs):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        save_path = out_dir / f"fig_show_{num}.png"
        fig.savefig(save_path, bbox_inches="tight")
plt.show = patched_plt_show

namespace = {}
try:
    sys.stdout = buf
    sys.stderr = err
    namespace = runpy.run_path("EDA_converted.py", run_name="__main__")
except Exception:
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr
    st.error("Error running EDA_converted.py")
    st.code(traceback.format_exc())
    raise SystemExit()
finally:
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr

# Restore originals
plt.show = _original_plt_show
if pio:
    pio.show = original_plotly_show

# ---------- OUTPUT TEXT ----------
output_text = buf.getvalue().strip()
if output_text:
    st.subheader("Console Output / Insights")
    st.code(output_text)

# ---------- MATPLOTLIB FIGURES ----------
saved_imgs = sorted(out_dir.glob("*.png"))
if saved_imgs:
    st.subheader("Matplotlib Figures")
    for img in saved_imgs:
        st.image(str(img), caption=img.name, use_column_width=True)

# ---------- PLOTLY CAPTURE ----------
if captured_plotly:
    st.subheader("Plotly Figures (Captured)")
    for fig in captured_plotly:
        st.plotly_chart(fig, use_container_width=True)

# ---------- ALTAR CHARTS ----------
if has_altair:
    alt_charts = [(name, val) for name, val in namespace.items() if isinstance(val, alt.Chart)]
    if alt_charts:
        st.subheader("Altair Charts")
        for name, chart in alt_charts:
            st.altair_chart(chart, use_container_width=True)
