# --- Replace Grad-CAM block with this improved, aligned version ---
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

def plot_mfcc_gradcam_glossy_aligned(fixed, cam, audio, sr,
                                     title="Feature map (MFCC + deltas)",
                                     mfcc_cmap=PALETTE["mfcc_cmap"],
                                     cam_cmap=PALETTE["cam_cmap"],
                                     bg_color=PALETTE["bg"],
                                     fg_color=PALETTE["fg"],
                                     muted=PALETTE["muted"]):
    """
    Improved Grad-CAM + MFCC glossy plotting with precise alignment.
    - fixed: (T_frames, F_bins)
    - cam:   (T_frames,)
    - audio, sr: used to compute duration for x-axis (seconds)
    Returns matplotlib.figure.Figure
    """
    T_frames = fixed.shape[0]
    F_bins = fixed.shape[1]
    duration_s = len(audio) / sr
    # convert frame index -> seconds mapping
    time_coords = np.linspace(0.0, duration_s, T_frames)

    # Figure and GridSpec: two rows, one narrow for cam, one larger for mfcc, colorbar on right
    fig = plt.figure(figsize=(12, 4.2), dpi=120, facecolor="none")
    fig.patch.set_alpha(0)

    gs = gridspec.GridSpec(nrows=2, ncols=10, figure=fig, left=0.06, right=0.92, top=0.95, bottom=0.10,
                           hspace=0.12, wspace=0.02)
    ax_mfcc = fig.add_subplot(gs[0, 0:9])  # main MFCC (row 0, cols 0-8)
    ax_cam  = fig.add_subplot(gs[1, 0:9], sharex=ax_mfcc)  # CAM below it (row1, cols 0-8)
    cax = fig.add_subplot(gs[:, 9])  # colorbar (full height of both rows)

    # Glossy rounded card backdrop in figure coords (behind axes)
    rect_bg = patches.FancyBboxPatch(
        (0.02, 0.06), 0.90, 0.88,
        transform=fig.transFigure,
        boxstyle="round,pad=0.02,rounding_size=12",
        linewidth=0,
        facecolor=(0.04, 0.04, 0.05, 0.72),
        zorder=0
    )
    fig.patches.append(rect_bg)

    # Plot MFCC heatmap with extent in seconds -> (xmin, xmax, ymin, ymax)
    im = ax_mfcc.imshow(
        fixed.T,
        origin="lower",
        aspect="auto",
        cmap=mfcc_cmap,
        interpolation='nearest',
        norm=Normalize(vmin=np.percentile(fixed, 5), vmax=np.percentile(fixed, 99)),
        extent=[0.0, duration_s, 0, F_bins]
    )
    ax_mfcc.set_ylabel("Feature bins", color=muted, fontsize=10)
    ax_mfcc.set_yticks(np.linspace(0, F_bins - 1, min(6, F_bins)).astype(int))
    # x ticks shown only on lower axis (we'll hide top)
    ax_mfcc.set_xticklabels([])
    ax_mfcc.tick_params(axis='y', colors=muted)
    ax_mfcc.set_title(title, color=fg_color, fontsize=12, pad=8, weight='600')

    # CAM heatstrip: tile cam vertically, plot with same x-extent (seconds)
    cam_map = np.tile(cam, (max(3, F_bins//8), 1))  # make strip visible even for few bins
    ax_cam.imshow(
        cam_map,
        origin='lower',
        aspect='auto',
        cmap=cam_cmap,
        interpolation='bilinear',
        extent=[0.0, duration_s, 0, 1]  # normalized vertical extent for the strip
    )
    ax_cam.set_xlabel("Time (s)", color=muted, fontsize=10)
    ax_cam.set_yticks([])  # no vertical ticks on CAM strip
    ax_cam.tick_params(axis='x', colors=muted)

    # Colorbar aligned to MFCC scale (use the MFCC image)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical', pad=0.01)
    cbar.set_label("Feature magnitude", color=muted, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=muted)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=muted)

    # Gridlines and thin frames for a technical look
    ax_mfcc.grid(axis='x', color='#0b0c0d', linestyle='--', linewidth=0.35, alpha=0.45)
    ax_cam.grid(axis='x', color='#0b0c0d', linestyle='--', linewidth=0.35, alpha=0.45)
    for spine in ax_mfcc.spines.values():
        spine.set_edgecolor("#121314")
        spine.set_linewidth(0.6)
    for spine in ax_cam.spines.values():
        spine.set_edgecolor("#121314")
        spine.set_linewidth(0.6)

    # Make x-ticks for both axes consistent (in seconds)
    n_ticks = 5
    xticks = np.linspace(0.0, duration_s, n_ticks)
    xtick_labels = [f"{t:.2f}s" for t in xticks]
    ax_cam.set_xticks(xticks)
    ax_cam.set_xticklabels(xtick_labels, color=muted, fontsize=9)
    # ensure ax_mfcc uses same xlim (no rounding mismatch)
    ax_mfcc.set_xlim(0.0, duration_s)
    ax_cam.set_xlim(0.0, duration_s)

    # Neon vertical marker: convert frame index -> seconds and draw on both axes
    peak_frame_idx = int(np.argmax(cam))
    peak_time = float((peak_frame_idx / max(1, T_frames - 1)) * duration_s)
    for a in (ax_mfcc, ax_cam):
        a.axvline(x=peak_time, color=PALETTE["neon"], linewidth=1.1, alpha=0.95, zorder=3)

    # Small info box inside figure coords (top-right inside the glossy card)
    info_x, info_y = 0.70, 0.82
    info_box = patches.FancyBboxPatch(
        (info_x, info_y), 0.26, 0.12,
        transform=fig.transFigure,
        boxstyle="round,pad=0.02,rounding_size=8",
        linewidth=0.5,
        edgecolor=(1,1,1,0.03),
        facecolor=(1,1,1,0.015),
        zorder=4
    )
    fig.patches.append(info_box)
    fig.text(info_x + 0.02, info_y + 0.06, "Activation", color=muted, fontsize=9)
    fig.text(info_x + 0.02, info_y + 0.02, f"Peak: {peak_frame_idx} ({peak_time:.2f}s)", color=fg_color, fontsize=10, weight='600')

    # Tighten layout and return
    plt.subplots_adjust(left=0.06, right=0.92, top=0.95, bottom=0.10, hspace=0.12)
    return fig

# --- Usage: replace your old Grad-CAM plotting call with this ---
try:
    fig = plot_mfcc_gradcam_glossy_aligned(fixed, cam, y, sr,
                                          title="Feature map (MFCC + deltas)",
                                          mfcc_cmap=PALETTE["mfcc_cmap"],
                                          cam_cmap=PALETTE["cam_cmap"])
    st.markdown("<div class='label-small'>NEURAL ACTIVATION MAP [GRAD-CAM]</div>", unsafe_allow_html=True)
    st.pyplot(fig)
    plt.close(fig)
except Exception as e:
    st.error(f"Grad-CAM failed: {e}")
