
import runpy, sys, io, traceback, types
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="EDA Visuals", layout="wide")
st.title("EDA Visuals (from EDA.ipynb) — FULL Graph view")


# --- Render notebook markdown insights (if EDA.ipynb exists) ---
try:
    import nbformat
    nb_path = Path("EDA.ipynb")
    if nb_path.exists():
        nb = nbformat.read(str(nb_path), as_version=4)
        md_cells = [cell for cell in nb.cells if cell.cell_type == "markdown"]
        if md_cells:
            st.sidebar.header("Notebook insights / Markdown")
            for i, cell in enumerate(md_cells, 1):
                # display markdown in the main page as collapsible sections in the sidebar
                try:
                    st.sidebar.markdown(f\"**Section {i}**\")
                    st.sidebar.markdown(cell.source)
                except Exception:
                    pass
except Exception:
    # if nbformat missing or read fails, ignore silently
    pass

out_dir = Path("plots_streamlit")
out_dir.mkdir(exist_ok=True)

# If dataset missing, show uploader (same as previous)
expected_xlsx = Path("final_cleaned_zomato_data.xlsx")
expected_csv = Path("final_cleaned_zomato_data.csv")
if not expected_xlsx.exists() and not expected_csv.exists():
    st.warning("Dataset file not found in the app folder. Please upload your dataset (Excel or CSV).")
    uploaded = st.file_uploader("Upload final_cleaned_zomato_data.xlsx or .csv", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        st.info("Upload the dataset to proceed. After uploading, the app will run the EDA and display plots.")
        st.stop()
    else:
        suffix = Path(uploaded.name).suffix.lower()
        save_path = expected_xlsx if suffix in [".xlsx", ".xls"] else expected_csv
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved uploaded file to {save_path}")

# Capture stdout/stderr fully
buf = io.StringIO()
err = io.StringIO()
sys_stdout = sys.stdout
sys_stderr = sys.stderr

# Prepare containers to capture plotly shows
captured_plotly = []

# Patch matplotlib.pyplot.show to save figures instead of depending on interactive backend
_original_plt_show = plt.show
def _patched_plt_show(*args, **kwargs):
    # save all current figures
    for i, num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(num)
        out_file = out_dir / f"matplotlib_shown_{num}.png"
        try:
            fig.savefig(out_file, bbox_inches='tight')
        except Exception:
            pass
    # do not call original show to avoid blocking
plt.show = _patched_plt_show

# Try to patch plotly.io.show so fig.show() and plotly.io.show(fig) are captured
try:
    import plotly.io as pio
    def _patched_plotly_show(fig, *args, **kwargs):
        try:
            # store figure object for later display
            captured_plotly.append(fig)
        except Exception:
            pass
    pio_show_original = getattr(pio, 'show', None)
    pio.show = _patched_plotly_show
    # Also patch plotly.graph_objs.Figure.show if available
    try:
        from plotly.graph_objs import Figure as PlotlyFigure
        orig_fig_show = PlotlyFigure.show
        def _fig_show(self, *args, **kwargs):
            try:
                captured_plotly.append(self)
            except Exception:
                pass
        PlotlyFigure.show = _fig_show
    except Exception:
        pass
except Exception:
    pio = None

# Try to import altair for detection later
try:
    import altair as alt
    has_altair = True
except Exception:
    alt = None
    has_altair = False

namespace = {}
try:
    sys.stdout = buf
    sys.stderr = err
    namespace = runpy.run_path("EDA_converted.py", run_name="__main__")
except Exception:
    tb = traceback.format_exc()
    # Restore stdout before showing error
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr
    st.error("Error while executing EDA_converted.py")
    st.code(tb)
    raise SystemExit()  # stop further execution of wrapper
finally:
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr

# Restore patched functions to originals where possible (avoid side effects)
try:
    plt.show = _original_plt_show
except Exception:
    pass
try:
    if pio and pio_show_original is not None:
        pio.show = pio_show_original
except Exception:
    pass

# Show full textual output (no truncation)
output_text = buf.getvalue().strip()
if output_text:
    st.subheader("Captured textual output")
    st.code(output_text)

# Collect matplotlib figures saved by patched plt.show or existing figures
saved_images = sorted(out_dir.glob("matplotlib_*.png")) + sorted(out_dir.glob("matplotlib_shown_*.png"))
fig_nums = plt.get_fignums()
if fig_nums or saved_images:
    st.subheader("Matplotlib plots")
    # First display saved images
    for p in saved_images:
        st.image(str(p), caption=p.name, use_column_width=True)
    # Then any remaining live figures
    for i, num in enumerate(fig_nums, 1):
        try:
            fig = plt.figure(num)
            out_file = out_dir / f"matlive_{num}.png"
            fig.savefig(out_file, bbox_inches='tight')
            st.image(str(out_file), caption=f"Matplotlib live figure {num}", use_column_width=True)
        except Exception:
            pass

# Display any captured plotly figures (from fig.show() or plotly.io.show)
if captured_plotly:
    st.subheader("Captured Plotly figures (from fig.show / plotly.io.show)")
    for i, fig in enumerate(captured_plotly, 1):
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # fallback: try to write to png if kaleido present
            try:
                out_file = out_dir / f"plotly_captured_{i}.png"
                fig.write_image(str(out_file))
                st.image(str(out_file), caption=f"Plotly captured {i}", use_column_width=True)
            except Exception:
                st.warning(f"Could not render captured Plotly figure #{i}")

# Display plotly figures assigned to variables in namespace
try:
    import plotly.graph_objs as go
    from plotly.graph_objs import Figure as PlotlyFigure
    plotly_in_namespace = [(n,v) for n,v in namespace.items() if isinstance(v, PlotlyFigure)]
    if plotly_in_namespace:
        st.subheader("Plotly figures assigned to variables")
        for name, fig in plotly_in_namespace:
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                try:
                    out_file = out_dir / f"plotly_var_{name}.png"
                    fig.write_image(str(out_file))
                    st.image(str(out_file), caption=f"Plotly var {name}", use_column_width=True)
                except Exception:
                    st.warning(f"Could not render Plotly figure '{name}'")
except Exception:
    pass

# Display Altair charts assigned to variables
if has_altair:
    alt_found = []
    for name, val in namespace.items():
        try:
            if isinstance(val, alt.Chart):
                alt_found.append((name, val))
        except Exception:
            pass
    if alt_found:
        st.subheader("Altair charts")
        for name, chart in alt_found:
            try:
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.warning(f"Could not render Altair chart '{name}'")

# Final message if nothing found
if not (saved_images or plt.get_fignums() or captured_plotly or (has_altair and any(isinstance(v, alt.Chart) for v in namespace.values()))):
    st.info("No plots were detected. If your notebook uses interactive backends or prints figures without saving or assigning them, edit EDA_converted.py to save figures using plt.savefig(...) or assign Plotly/Altair charts to variables.")
