# PyMOL script to generate comparison figures (v3)

# Load the structures
load 1aon_tm_fft_v4_rmsd_4.81.pdb, our_result
load 1GRU.cif, gold_standard

# Style the representations
show cartoon
color gray80, gold_standard
# Color our result by chain for better visualization
util.cbag our_result

# Set general appearance
bg_color white
set ray_trace_frames, 1
set antialias, 2
viewport 800, 600

# Center the view on the loaded objects without aligning them
center all
zoom all

# --- View 1: Frontal ---
# Capture the first view
ray
png figure_1A.png, dpi=300, quiet=0

# --- View 2: Rotated 90 degrees ---
# Rotate 90 degrees around the Y axis and re-render
rotate y, 90
ray
png figure_1B.png, dpi=300, quiet=0

quit