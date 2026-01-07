import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from functools import partial
import argparse

# I should probably use some actual audio library like librosa
# but this is numpy + matplotlib and google helping me

# --- Constants ---
CHUNK_DURATION_SEC = 2.0
STEP_DURATION_SEC = 1.0
BYTES_PER_SAMPLE = 4
MAX_Y_LIMIT_HARD = 1.0  # Hard Limit for full scale unity
MIN_Y_LIMIT_HARD = -1.0
MIN_Y_ZOOM_DEFAULT = 0.5  # Default minimum Y view range (-0.5 to 0.5)

# --- Global State ---
file_handles = []
current_offset_bytes = 0
current_chunk_samples = 0
SAMPLE_RATE_HZ = 48000

def zoom_handler(event, ax):
    """Handles X/Y scroll zoom centered on cursor, constrained to current 4s chunk."""
    if event.inaxes is None: return

    scale_factor = 0.8 if event.button == 'up' else 1.2

    # X-Axis limits (0 to CHUNK_DURATION_SEC)
    xlim = ax.get_xlim()
    cur_width = xlim[1] - xlim[0]
    new_width = cur_width * scale_factor

    # arbitrary X limit: 100 samples
    new_width = max(100 / SAMPLE_RATE_HZ, new_width)

    rel_pos_x = (event.xdata - xlim[0]) / cur_width if cur_width > 0 else 0.5
    new_left = event.xdata - (new_width * rel_pos_x)
    new_right = new_left + new_width

    ax.set_xlim([max(0, new_left), min(CHUNK_DURATION_SEC, new_right)])

    # Y-Axis limits (-1.0 to 1.0)
    ylim = ax.get_ylim()
    cur_height = ylim[1] - ylim[0]
    new_height = cur_height * scale_factor

    # arbitrary Y limit: 1/10th of the scale
    new_height = max(0.2, new_height)

    rel_pos_y = (event.ydata - ylim[0]) / cur_height if cur_height > 0 else 0.5
    new_bottom = event.ydata - (new_height * rel_pos_y)
    new_top = new_bottom + new_height

    # Constrain Y to hard limits
    ax.set_ylim([max(MIN_Y_LIMIT_HARD, new_bottom), min(MAX_Y_LIMIT_HARD, new_top)])
    ax.figure.tight_layout()
    ax.figure.canvas.draw_idle()

def update_plot_chunk(direction, ax, fig):
    """
    Moves file pointer by 2s, reads a 4s chunk, and sets the Y-axis
    based on the *actual* data range within that chunk.
    """
    global current_offset_bytes, current_chunk_samples

    step_bytes = int(STEP_DURATION_SEC * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE)
    new_offset = current_offset_bytes + (direction * step_bytes)

    if new_offset < 0: new_offset = 0
    current_offset_bytes = new_offset

    ax.clear()
    read_count = int(CHUNK_DURATION_SEC * SAMPLE_RATE_HZ)

    # Track the min/max of all loaded data for setting Y-axis limits
    global_min_y, global_max_y = 1e9, -1e9

    for handle, name in file_handles:
        handle.seek(current_offset_bytes, os.SEEK_SET)
        data = np.fromfile(handle, dtype=np.int32, count=read_count)

        if data.size > 0:
            normalized = data / 2147483648.0
            time_axis = np.arange(data.size) / SAMPLE_RATE_HZ
            ax.plot(time_axis, normalized, linewidth=0.8, label=name)
            current_chunk_samples = data.size

            # Update the global min/max for this specific loaded chunk
            global_min_y = min(global_min_y, normalized.min())
            global_max_y = max(global_max_y, normalized.max())

    # Absolute time formatting
    abs_offset_samples = current_offset_bytes // BYTES_PER_SAMPLE
    def time_fmt(x, pos):
        return f'{(x + abs_offset_samples / SAMPLE_RATE_HZ):.3f}s'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))

    # --- Y-Axis Auto-Ranging Logic ---
    if global_max_y > global_min_y:
        # Calculate a symmetric range around zero that encompasses the data
        max_abs = max(abs(global_min_y), abs(global_max_y))

        # Default to a range of at least -0.5 to 0.5 (MIN_Y_ZOOM_DEFAULT)
        # We ensure the limits are at least as wide as our minimum default zoom level
        # and do not exceed the hard -1.0 to 1.0 limits.
        view_limit = max(max_abs, MIN_Y_ZOOM_DEFAULT)

        # Add a little padding to the view_limit
        view_limit *= 1.05

        # Clamp the view limits to the hard limits (-1.0 to 1.0)
        view_limit = min(view_limit, MAX_Y_LIMIT_HARD)

        ax.set_ylim(-view_limit, view_limit)
    else:
        # Fallback if chunk is empty or flatlined
        ax.set_ylim(-MIN_Y_ZOOM_DEFAULT, MIN_Y_ZOOM_DEFAULT)


    # Reset X View (always 0 to 4s for this logic)
    ax.set_xlim(0, CHUNK_DURATION_SEC)
    ax.axhline(0, color='black', lw=0.5, ls='--')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.set_title(f"Position: {(abs_offset_samples/SAMPLE_RATE_HZ):.3f}s | Fs: {SAMPLE_RATE_HZ}Hz")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize='x-small')

    fig.tight_layout()
    fig.canvas.draw_idle()

def setup_app(filenames, rate):
    global file_handles, SAMPLE_RATE_HZ
    SAMPLE_RATE_HZ = rate

    for f in filenames:
        try:
            file_handles.append((open(f, 'rb'), os.path.basename(f)))
        except Exception as e:
            print(f"Error opening {f}: {e}")

    if not file_handles: return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Event Connections
    fig.canvas.mpl_connect('scroll_event', partial(zoom_handler, ax=ax))
    fig.canvas.mpl_connect('key_press_event', lambda e:
        update_plot_chunk(1, ax, fig) if e.key == 'right' else
        update_plot_chunk(-1, ax, fig) if e.key == 'left' else None)

    # Initial Render
    update_plot_chunk(0, ax, fig)

    # Enable "Zoom to Rectangle" by default
    try:
        fig.canvas.toolbar.zoom()
    except:
        pass

    plt.show()
    for h, _ in file_handles: h.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linux Audio Waveform Visualizer 2026")
    parser.add_argument('files', nargs='+', help="Input .bin files (int32)")
    parser.add_argument('--rate', type=int, default=48000, help="Sample rate (Hz)")
    args = parser.parse_args()
    setup_app(args.files, args.rate)

