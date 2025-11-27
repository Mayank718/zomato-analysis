import runpy, sys, io, traceback
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="EDA Visuals", layout="wide")
st.title("EDA Visuals (from EDA.ipynb) — Graph view")

out_dir = Path("plots_streamlit")
out_dir.mkdir(exist_ok=True)

buf = io.StringIO()
err = io.StringIO()
sys_stdout = sys.stdout
sys_stderr = sys.stderr

has_plotly = False
has_altair = False
try:
    import plotly.graph_objs as go
    from plotly.graph_objs import Figure as PlotlyFigure
    has_plotly = True
except Exception:
    PlotlyFigure = None
try:
    import altair as alt
    has_altair = True
except Exception:
    alt = None

namespace = {}
try:
    sys.stdout = buf
    sys.stderr = err
    namespace = runpy.run_path("EDA_converted.py", run_name="__main__")
except Exception:
    st.error("Error executing EDA_converted.py")
    st.code(traceback.format_exc())
finally:
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr

output = buf.getvalue().strip()
if output:
    st.subheader("Output")
    st.code(output[:1500] + ("... truncated" if len(output)>1500 else ""))

# Matplotlib
fig_nums = plt.get_fignums()
if fig_nums:
    st.subheader("Matplotlib Figures")
    for i, num in enumerate(fig_nums,1):
        fig = plt.figure(num)
        path = out_dir/f"matplotlib_{i}.png"
        fig.savefig(path, bbox_inches="tight")
        st.image(str(path), use_column_width=True)

# Plotly
if has_plotly:
    for k,v in namespace.items():
        if isinstance(v, PlotlyFigure):
            st.subheader(f"Plotly: {k}")
            st.plotly_chart(v, use_container_width=True)

# Altair
if has_altair:
    for k,v in namespace.items():
        if isinstance(v, alt.Chart):
            st.subheader(f"Altair: {k}")
            st.altair_chart(v, use_container_width=True)
