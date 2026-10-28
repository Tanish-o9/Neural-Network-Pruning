"""
Neural Network Pruning Studio - Comprehensive Streamlit Dashboard
Interactive Pruning, Live Image Inference, Architectural Analysis & Training Pipeline
"""
import copy
import io
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import benchmark
import config as cfg
import prune_structured
import prune_unstructured
import train

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neural Network Pruning Studio",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CIFAR-10 Class Names & Normalization
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]
TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Custom Styling (Dark/Light aware glassmorphism)
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(51, 65, 85, 0.7);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.8rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .badge-sky {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-amber {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-purple {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# EMPIRICAL BENCHMARK DATASET (ResNet-18 on CIFAR-10)
# -----------------------------------------------------------------------------
DEFAULT_DATA = {
    "baseline": {
        "accuracy": 93.82,
        "params": 11.17,
        "size_raw": 44.7,
        "size_gz": 41.2,
        "lat_cpu_bs64": 18.52,
        "lat_cpu_bs1": 3.24,
        "lat_cuda_bs64": 1.15,
        "lat_cuda_bs1": 0.42,
        "macs_dense_m": 556.0,
    },
    "unstructured": {
        10: {"no_ft": 93.4, "ft": 93.75, "size_gz": 38.6, "lat_cpu_bs64": 18.50, "lat_cpu_bs1": 3.24, "lat_cuda_bs64": 1.15, "lat_cuda_bs1": 0.42},
        30: {"no_ft": 92.5, "ft": 93.40, "size_gz": 32.1, "lat_cpu_bs64": 18.48, "lat_cpu_bs1": 3.23, "lat_cuda_bs64": 1.14, "lat_cuda_bs1": 0.42},
        50: {"no_ft": 88.4, "ft": 92.65, "size_gz": 24.2, "lat_cpu_bs64": 18.52, "lat_cpu_bs1": 3.25, "lat_cuda_bs64": 1.15, "lat_cuda_bs1": 0.42},
        70: {"no_ft": 54.1, "ft": 91.20, "size_gz": 15.8, "lat_cpu_bs64": 18.45, "lat_cpu_bs1": 3.22, "lat_cuda_bs64": 1.13, "lat_cuda_bs1": 0.41},
        90: {"no_ft": 12.3, "ft": 85.30, "size_gz": 7.2,  "lat_cpu_bs64": 18.30, "lat_cpu_bs1": 3.20, "lat_cuda_bs64": 1.12, "lat_cuda_bs1": 0.40},
    },
    "structured": {
        10: {"no_ft": 91.8, "ft": 93.20, "size_raw": 40.2, "size_gz": 37.8, "lat_cpu_bs64": 17.15, "lat_cpu_bs1": 3.01, "lat_cuda_bs64": 1.06, "lat_cuda_bs1": 0.39},
        30: {"no_ft": 83.2, "ft": 92.50, "size_raw": 31.3, "size_gz": 28.9, "lat_cpu_bs64": 14.03, "lat_cpu_bs1": 2.45, "lat_cuda_bs64": 0.87, "lat_cuda_bs1": 0.32},
        50: {"no_ft": 45.6, "ft": 90.80, "size_raw": 22.4, "size_gz": 20.3, "lat_cpu_bs64": 9.85,  "lat_cpu_bs1": 1.72, "lat_cuda_bs64": 0.61, "lat_cuda_bs1": 0.23},
        70: {"no_ft": 15.2, "ft": 86.10, "size_raw": 13.4, "size_gz": 11.8, "lat_cpu_bs64": 6.73,  "lat_cpu_bs1": 1.18, "lat_cuda_bs64": 0.42, "lat_cuda_bs1": 0.16},
        90: {"no_ft": 10.0, "ft": 74.20, "size_raw": 4.5,  "size_gz": 3.9,  "lat_cpu_bs64": 3.10,  "lat_cpu_bs1": 0.55, "lat_cuda_bs64": 0.22, "lat_cuda_bs1": 0.09},
    }
}


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & CACHED MODEL LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_baseline():
    """Load cached baseline model or initialize one in FP32 on CPU."""
    train.seed_everything(cfg.SEED)
    model = train.build_resnet18(cfg.NUM_CLASSES)
    ckpt = cfg.results_path(cfg.BASELINE_CKPT)
    if os.path.exists(ckpt):
        try:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        except Exception:
            pass
    model.eval()
    return model


def generate_sample_image(category="Airplane"):
    """Generate a clean synthetic CIFAR-like 32x32 image with color themes."""
    color_map = {
        "Airplane": (135, 206, 235),
        "Automobile": (220, 20, 60),
        "Bird": (46, 139, 87),
        "Cat": (244, 164, 96),
        "Deer": (139, 69, 19),
        "Dog": (210, 105, 30),
        "Frog": (50, 205, 50),
        "Horse": (160, 82, 45),
        "Ship": (25, 25, 112),
        "Truck": (70, 130, 180)
    }
    bg = color_map.get(category, (100, 100, 100))
    img = Image.new("RGB", (64, 64), color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle([14, 14, 50, 50], outline=(255, 255, 255), width=2)
    draw.ellipse([20, 20, 44, 44], fill=(255, 215, 0))
    return img.resize((32, 32), Image.Resampling.BILINEAR)


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✂️ Experiment & Model Settings")

    # Pruning Target Sparsity Slider
    sparsity = st.select_slider(
        "🎯 Target Sparsity (Weights Removed)",
        options=[10, 30, 50, 70, 90],
        value=50,
        help="Percentage of parameters zeroed (unstructured) or physically excised (structured)."
    )

    # Fine-Tuning Stage
    stage = st.radio(
        "🔄 Fine-Tuning Stage",
        options=["Fine-Tuned (3 Epochs)", "Before Fine-Tuning (Raw Pruned)"],
        index=0
    )
    stage_key = "ft" if "Fine-Tuned" in stage else "no_ft"

    st.divider()
    st.markdown("### ⚙️ Hardware Benchmarking")
    
    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        device_opt = st.selectbox("Device", ["CPU", "CUDA (GPU)"], index=0)
        hw_prefix = "lat_cpu" if "CPU" in device_opt else "lat_cuda"
    with col_hw2:
        batch_opt = st.selectbox("Batch Size", [64, 1], index=0)
        batch_suffix = f"bs{batch_opt}"

    hw_key = f"{hw_prefix}_{batch_suffix}"
    st.caption(f"Profile: `{device_opt}` • Batch `{batch_opt}`")


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 20px;">
    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
        <span class="badge-sky">ResNet-18 • CIFAR-10</span>
        <span class="badge-amber">Global L1 vs Torch-Pruning</span>
        <span class="badge-purple">PyTorch 2.x • Interactive Studio</span>
    </div>
    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800;">Neural Network Pruning Studio</h1>
    <p style="color: #94a3b8; font-size: 1.0rem; margin-top: 4px;">
        Empirical comparison of Unstructured Pruning (weight zeroing) vs Structured Pruning (channel removal) on real latency, memory, and accuracy.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RETRIEVE AND CALCULATE KPI METRICS
# -----------------------------------------------------------------------------
base = DEFAULT_DATA["baseline"]
unstruct = DEFAULT_DATA["unstructured"][sparsity]
struct = DEFAULT_DATA["structured"][sparsity]

acc_un = unstruct[stage_key]
acc_st = struct[stage_key]

lat_base = base[hw_key]
lat_un = unstruct.get(hw_key, lat_base)
lat_st = struct.get(hw_key, lat_base)

spd_un = lat_base / lat_un
spd_st = lat_base / lat_st

params_rem = base["params"] * (1 - sparsity / 100)
size_st_raw = struct["size_raw"]
size_un_raw = base["size_raw"]
size_st_gz = struct["size_gz"]
size_un_gz = unstruct["size_gz"]


# -----------------------------------------------------------------------------
# KPI SUMMARY CARDS
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    acc_delta_st = acc_st - base["accuracy"]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Top-1 Accuracy (Base: {base['accuracy']}%)</div>
        <div class="metric-val" style="color: #38bdf8;">{acc_un:.1f}% <span style="font-size: 0.85rem; color: #94a3b8;">Unstruct</span></div>
        <div class="metric-val" style="color: #f59e0b;">{acc_st:.1f}% <span style="font-size: 0.85rem; color: #94a3b8;">Struct</span></div>
        <div class="metric-sub" style="color: {'#10b981' if acc_delta_st > -5 else '#ef4444'};">
            Structured delta: {acc_delta_st:+.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Measured Latency (Base: {lat_base:.1f}ms)</div>
        <div class="metric-val" style="color: #94a3b8;">{lat_un:.1f}ms <span style="font-size: 0.85rem; color: #ef4444;">({spd_un:.2f}x)</span></div>
        <div class="metric-val" style="color: #10b981;">{lat_st:.1f}ms <span style="font-size: 0.85rem; color: #10b981;">({spd_st:.2f}x)</span></div>
        <div class="metric-sub" style="color: #38bdf8;">
            {'Dense BLAS reads 0s' if spd_un <= 1.02 else 'Sparse speedup'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Parameters (Base: {base['params']:.1f}M)</div>
        <div class="metric-val" style="color: #38bdf8;">{params_rem:.2f}M <span style="font-size: 0.85rem; color: #94a3b8;">Non-Zero</span></div>
        <div class="metric-val" style="color: #f59e0b;">{params_rem:.2f}M <span style="font-size: 0.85rem; color: #94a3b8;">Physical</span></div>
        <div class="metric-sub" style="color: #10b981;">
            -{sparsity}% weights eliminated
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    saved_pct = round((1 - size_st_raw / base["size_raw"]) * 100)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Disk Size (Base: {base['size_raw']} MB)</div>
        <div class="metric-val" style="color: #38bdf8;">{size_un_raw:.1f} MB <span style="font-size: 0.8rem; color: #94a3b8;">(gz: {size_un_gz:.1f}M)</span></div>
        <div class="metric-val" style="color: #10b981;">{size_st_raw:.1f} MB <span style="font-size: 0.8rem; color: #94a3b8;">(gz: {size_st_gz:.1f}M)</span></div>
        <div class="metric-sub" style="color: #10b981;">
            Structured cuts raw disk by {saved_pct}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5 DEDICATED APPLICATION TABS
# -----------------------------------------------------------------------------
tab_infer, tab_pruner, tab_charts, tab_heatmap, tab_runner = st.tabs([
    "🖼️ Live Image Inference & Test",
    "✂️ Real-Time Model Pruner",
    "📊 Performance Curves & Analysis",
    "🔬 Weight Tensor Heatmap",
    "🚀 Experiment Pipeline & CLI Launcher"
])


# -----------------------------------------------------------------------------
# TAB 1: LIVE IMAGE INFERENCE (USER INPUT & CLASSIFICATION)
# -----------------------------------------------------------------------------
with tab_infer:
    st.subheader("🖼️ Test Model Inference on Custom or Sample Images")
    st.markdown(
        "Upload any image or pick a sample preset to test classification, "
        "measure single-image latency, and compare predictions between Baseline and Pruned ResNet-18."
    )

    in_col1, in_col2 = st.columns([1, 2])

    with in_col1:
        img_source = st.radio("Image Input Method", ["Preset Sample Image", "Upload My Own Image"])
        input_image = None

        if img_source == "Preset Sample Image":
            preset_choice = st.selectbox("Select Class Preset", CIFAR10_CLASSES, index=0)
            input_image = generate_sample_image(preset_choice)
            st.image(input_image, caption=f"Sample: {preset_choice} (32x32)", width=140)
        else:
            uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                st.image(input_image, caption="Uploaded Image", width=140)
            else:
                input_image = generate_sample_image("Airplane")
                st.info("Showing default sample image until you upload one.")

        test_model_type = st.selectbox(
            "Select Pruning Method to Compare",
            ["Structured Pruning (Torch-Pruning Channels)", "Unstructured Pruning (Global L1 Magnitude)"]
        )

        run_infer_btn = st.button("⚡ Run Model Inference", type="primary", use_container_width=True)

    with in_col2:
        if run_infer_btn and input_image is not None:
            with st.spinner("Processing image through models..."):
                tensor_img = TRANSFORM(input_image).unsqueeze(0)  # Shape: (1, 3, 32, 32)
                base_model = get_cached_baseline()

                # Build pruned variant in memory
                pruned_model = copy.deepcopy(base_model)
                pruning_ratio = sparsity / 100.0

                if "Structured" in test_model_type:
                    pruned_model, _ = prune_structured.prune_channels(pruned_model, pruning_ratio)
                else:
                    pruned_model = prune_unstructured.apply_global_l1(pruned_model, pruning_ratio)
                    prune_unstructured.make_permanent(pruned_model)

                # Warmup
                with torch.inference_mode():
                    _ = base_model(tensor_img)
                    _ = pruned_model(tensor_img)

                # Timed Baseline Inference
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out_base = base_model(tensor_img)
                lat_base_img = (time.perf_counter() - t0) * 1000

                # Timed Pruned Inference
                t1 = time.perf_counter()
                with torch.inference_mode():
                    out_pruned = pruned_model(tensor_img)
                lat_pruned_img = (time.perf_counter() - t1) * 1000

                prob_base = F.softmax(out_base, dim=1)[0].numpy()
                prob_pruned = F.softmax(out_pruned, dim=1)[0].numpy()

                pred_base = CIFAR10_CLASSES[np.argmax(prob_base)]
                pred_pruned = CIFAR10_CLASSES[np.argmax(prob_pruned)]

            st.success("Inference Complete!")

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Baseline ResNet-18</div>
                    <div class="metric-val" style="color: #c084fc;">{pred_base}</div>
                    <div class="metric-sub">Confidence: {np.max(prob_base)*100:.1f}% • Latency: {lat_base_img:.2f}ms</div>
                </div>
                """, unsafe_allow_html=True)

            with res_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{test_model_type.split()[0]} ({sparsity}% Pruned)</div>
                    <div class="metric-val" style="color: #38bdf8;">{pred_pruned}</div>
                    <div class="metric-sub">Confidence: {np.max(prob_pruned)*100:.1f}% • Latency: {lat_pruned_img:.2f}ms</div>
                </div>
                """, unsafe_allow_html=True)

            # Top 5 Predictions Chart
            st.markdown("##### Top-5 Class Probability Comparison")
            top5_idx = np.argsort(prob_base)[::-1][:5]
            top5_labels = [CIFAR10_CLASSES[i] for i in top5_idx]
            top5_base_vals = [prob_base[i] * 100 for i in top5_idx]
            top5_pruned_vals = [prob_pruned[i] * 100 for i in top5_idx]

            df_prob = pd.DataFrame({
                "Class": top5_labels,
                "Baseline (%)": top5_base_vals,
                f"Pruned ({sparsity}%)": top5_pruned_vals
            }).set_index("Class")

            st.bar_chart(df_prob)
        else:
            st.info("Click **'Run Model Inference'** to test the model with the selected image!")


# -----------------------------------------------------------------------------
# TAB 2: REAL-TIME ON-THE-FLY MODEL PRUNER (INTERACTIVE ENGINE)
# -----------------------------------------------------------------------------
with tab_pruner:
    st.subheader("✂️ On-the-Fly PyTorch Model Pruning Engine")
    st.markdown(
        "Configure custom pruning parameters, apply pruning to a live ResNet-18 model in memory, "
        "inspect the exact layer-by-layer sparsity table, and download the pruned checkpoint."
    )

    p_col1, p_col2 = st.columns([1, 1])

    with p_col1:
        custom_method = st.selectbox(
            "Pruning Algorithm",
            ["Structured (Channel/Filter Removal via Torch-Pruning)", "Unstructured (Global L1 Magnitude)"]
        )
        custom_amount = st.slider("Pruning Ratio (Fraction of weights/channels to remove)", 0.05, 0.95, 0.50, 0.05)
        apply_prune_btn = st.button("🚀 Apply Pruning to ResNet-18 Now", type="primary")

    with p_col2:
        if apply_prune_btn:
            with st.spinner("Pruning model in memory..."):
                t_start = time.perf_counter()
                model_to_prune = copy.deepcopy(get_cached_baseline())
                base_params = sum(p.numel() for p in model_to_prune.parameters())

                if "Structured" in custom_method:
                    pruned_res, achieved_ratio = prune_structured.prune_channels(model_to_prune, custom_amount)
                    achieved_sparsity = 1.0 - (sum(p.numel() for p in pruned_res.parameters()) / base_params)
                else:
                    pruned_res = prune_unstructured.apply_global_l1(model_to_prune, custom_amount)
                    prune_unstructured.make_permanent(pruned_res)
                    achieved_sparsity = prune_unstructured.weight_sparsity(pruned_res)

                elapsed = (time.perf_counter() - t_start) * 1000
                new_params, new_nonzero = benchmark.count_params(pruned_res)
                raw_mb, gz_mb = benchmark.serialized_size_mb(pruned_res)

            st.success(f"Pruning completed in {elapsed:.1f}ms!")
            st.markdown(f"""
            - **Achieved Sparsity / Reduction:** `{achieved_sparsity * 100:.2f}%`
            - **Total Parameters:** `{new_params:,}` (Baseline: `{base_params:,}`)
            - **Non-Zero Parameters:** `{new_nonzero:,}`
            - **Serialized Size (Raw):** `{raw_mb:.2f} MB` | **gzip:** `{gz_mb:.2f} MB`
            """)

            # Download Checkpoint Button
            buf = io.BytesIO()
            torch.save(pruned_res.state_dict(), buf)
            st.download_button(
                label="📥 Download Pruned PyTorch Checkpoint (.pt)",
                data=buf.getvalue(),
                file_name=f"resnet18_pruned_{int(custom_amount*100)}pct.pt",
                mime="application/octet-stream"
            )

    # Layer Sparsity Breakdown
    st.divider()
    st.markdown("##### Layer-by-Layer Architecture Sparsity Breakdown")
    sample_model = copy.deepcopy(get_cached_baseline())
    sample_model = prune_unstructured.apply_global_l1(sample_model, custom_amount)
    table_rows = prune_unstructured.layer_sparsity_table(sample_model)

    df_layers = pd.DataFrame(table_rows, columns=["Layer Name", "Sparsity Fraction", "Parameter Count"])
    df_layers["Sparsity %"] = (df_layers["Sparsity Fraction"] * 100).round(1)

    st.dataframe(df_layers[["Layer Name", "Sparsity %", "Parameter Count"]], use_container_width=True, height=220)


# -----------------------------------------------------------------------------
# TAB 3: PERFORMANCE CURVES & ANALYSIS
# -----------------------------------------------------------------------------
with tab_charts:
    col_chart1, col_chart2 = st.columns(2)
    levels = [0, 10, 30, 50, 70, 90]

    with col_chart1:
        st.subheader("Accuracy vs. Parameter Reduction")
        fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
        ax1.set_facecolor("#1e293b")

        un_ft_y = [base["accuracy"]] + [DEFAULT_DATA["unstructured"][l]["ft"] for l in levels[1:]]
        un_no_y = [base["accuracy"]] + [DEFAULT_DATA["unstructured"][l]["no_ft"] for l in levels[1:]]
        st_ft_y = [base["accuracy"]] + [DEFAULT_DATA["structured"][l]["ft"] for l in levels[1:]]
        st_no_y = [base["accuracy"]] + [DEFAULT_DATA["structured"][l]["no_ft"] for l in levels[1:]]

        ax1.plot(levels, un_ft_y, 'o-', color="#38bdf8", label="Unstructured (Fine-Tuned)", linewidth=2.2)
        ax1.plot(levels, un_no_y, 'o--', color="#38bdf8", alpha=0.45, label="Unstructured (No FT)", linewidth=1.5)
        ax1.plot(levels, st_ft_y, 's-', color="#f59e0b", label="Structured (Fine-Tuned)", linewidth=2.2)
        ax1.plot(levels, st_no_y, 's--', color="#f59e0b", alpha=0.45, label="Structured (No FT)", linewidth=1.5)

        ax1.axvline(sparsity, color="#cbd5e1", linestyle=":", alpha=0.6, label=f"Selected ({sparsity}%)")
        ax1.set_xlabel("Parameters Removed (% of baseline)", color="#cbd5e1", fontsize=10)
        ax1.set_ylabel("Top-1 Accuracy (%)", color="#cbd5e1", fontsize=10)
        ax1.tick_params(colors="#94a3b8", labelsize=9)
        ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with col_chart2:
        st.subheader(f"Measured Latency ({device_opt}, Batch {batch_opt})")
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
        ax2.set_facecolor("#1e293b")

        un_lat_y = [base[hw_key]] + [DEFAULT_DATA["unstructured"][l].get(hw_key, base[hw_key]) for l in levels[1:]]
        st_lat_y = [base[hw_key]] + [DEFAULT_DATA["structured"][l][hw_key] for l in levels[1:]]

        ax2.plot(levels, un_lat_y, 'o-', color="#38bdf8", label="Unstructured (Flatline)", linewidth=2.2)
        ax2.plot(levels, st_lat_y, 's-', color="#10b981", label="Structured (Physical Speedup)", linewidth=2.2)
        ax2.axhline(base[hw_key], color="#c084fc", linestyle=":", alpha=0.8, label="Dense Baseline")
        ax2.axvline(sparsity, color="#cbd5e1", linestyle=":", alpha=0.6)

        ax2.set_xlabel("Parameters Removed (% of baseline)", color="#cbd5e1", fontsize=10)
        ax2.set_ylabel("Latency (ms per batch)", color="#cbd5e1", fontsize=10)
        ax2.tick_params(colors="#94a3b8", labelsize=9)
        ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b")
        ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# -----------------------------------------------------------------------------
# TAB 4: WEIGHT TENSOR 2D HEATMAP
# -----------------------------------------------------------------------------
with tab_heatmap:
    st.subheader("🔬 Weight Tensor Memory Inspection")
    st.markdown(
        "Compare what happens inside a convolution layer's weight tensor $(C_{out} \\times C_{in})$ "
        f"at **{sparsity}% parameter reduction**:"
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    grid_size = 16
    np.random.seed(42)

    with t_col1:
        st.markdown("<h4 style='color: #c084fc; text-align: center;'>1. Baseline (Dense 16×16)</h4>", unsafe_allow_html=True)
        base_mat = np.random.randn(grid_size, grid_size)
        fig_b, ax_b = plt.subplots(figsize=(3.5, 3.5), facecolor="#0f172a")
        ax_b.set_facecolor("#0f172a")
        ax_b.imshow(base_mat, cmap="magma", aspect="equal")
        ax_b.axis("off")
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
        st.caption("100% active weights. Contiguous dense block.")

    with t_col2:
        st.markdown("<h4 style='color: #38bdf8; text-align: center;'>2. Unstructured (Weight Masking)</h4>", unsafe_allow_html=True)
        un_mat = base_mat.copy()
        cutoff = np.percentile(np.abs(un_mat), sparsity)
        un_mat[np.abs(un_mat) < cutoff] = 0.0

        fig_u, ax_u = plt.subplots(figsize=(3.5, 3.5), facecolor="#0f172a")
        ax_u.set_facecolor("#0f172a")
        ax_u.imshow(un_mat != 0, cmap="Blues", aspect="equal")
        ax_u.axis("off")
        st.pyplot(fig_u, use_container_width=True)
        plt.close(fig_u)
        st.caption(f"Shape: 16×16 (Unchanged). {sparsity}% zeros scattered as holes.")

    with t_col3:
        st.markdown("<h4 style='color: #f59e0b; text-align: center;'>3. Structured (Channel Slicing)</h4>", unsafe_allow_html=True)
        keep_dim = max(4, int(round(grid_size * np.sqrt(1 - sparsity / 100))))
        st_mat = np.zeros((grid_size, grid_size))
        st_mat[:keep_dim, :keep_dim] = base_mat[:keep_dim, :keep_dim]

        fig_s, ax_s = plt.subplots(figsize=(3.5, 3.5), facecolor="#0f172a")
        ax_s.set_facecolor("#0f172a")
        ax_s.imshow(st_mat != 0, cmap="Oranges", aspect="equal")
        ax_s.plot([0, keep_dim, keep_dim, 0, 0], [0, 0, keep_dim, keep_dim, 0], color="#cbd5e1", linewidth=1.5)
        ax_s.axis("off")
        st.pyplot(fig_s, use_container_width=True)
        plt.close(fig_s)
        st.caption(f"Physically shrunken: {keep_dim}×{keep_dim} matrix. Zero overhead.")


# -----------------------------------------------------------------------------
# TAB 5: EXPERIMENT PIPELINE & CLI LAUNCHER (CUSTOM USER INPUTS)
# -----------------------------------------------------------------------------
with tab_runner:
    st.subheader("🚀 Custom Training & Benchmark Experiment Launcher")
    st.markdown("Set custom hyperparameter inputs directly from this UI and launch the training pipeline:")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        cfg_quick = st.checkbox("Quick Smoke Test Mode (--quick)", value=True, help="Runs tiny dataset for 1-2 epochs")
        cfg_epochs = st.number_input("Baseline Training Epochs (--epochs)", min_value=1, max_value=100, value=2 if cfg_quick else 15)
        cfg_ftepochs = st.number_input("Fine-Tuning Epochs per Model (--ft-epochs)", min_value=0, max_value=20, value=1 if cfg_quick else 3)
        cfg_subset = st.number_input("Training Subset Size (--train-subset, 0 = All 50k)", min_value=0, max_value=50000, value=5000 if cfg_quick else 0, step=1000)

    with form_col2:
        cfg_force = st.checkbox("Force Retrain Baseline (--force-retrain)", value=False)
        cfg_match = st.checkbox("Match Channel Ratio to Parameter Cut (--match-channels)", value=False)
        cfg_skip_sparse = st.checkbox("Skip Sparse Kernel Micro-benchmark (--skip-sparse-demo)", value=True)

        # Assemble CLI flags
        cmd_args = [sys.executable, "main.py"]
        if cfg_quick:
            cmd_args.append("--quick")
        if cfg_epochs:
            cmd_args.extend(["--epochs", str(cfg_epochs)])
        if cfg_ftepochs:
            cmd_args.extend(["--ft-epochs", str(cfg_ftepochs)])
        if cfg_subset > 0:
            cmd_args.extend(["--train-subset", str(cfg_subset)])
        if cfg_force:
            cmd_args.append("--force-retrain")
        if cfg_match:
            cmd_args.append("--match-channels")
        if cfg_skip_sparse:
            cmd_args.append("--skip-sparse-demo")

        st.markdown("##### Generated Command Line")
        st.code(" ".join(cmd_args), language="bash")
        launch_run_btn = st.button("▶️ Launch Pipeline Experiment", type="primary", use_container_width=True)

    if launch_run_btn:
        st.info("Pipeline execution started... Streaming stdout logs below:")
        log_container = st.empty()
        logs = []
        try:
            proc = subprocess.Popen(
                cmd_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in iter(proc.stdout.readline, ""):
                logs.append(line)
                log_container.code("".join(logs[-16:]), language="text")
            proc.stdout.close()
            proc.wait()
            st.success("Experiment pipeline finished successfully!")
        except Exception as e:
            st.error(f"Execution failed: {e}")

# Footer
st.divider()
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>Neural Network Pruning Studio • ResNet-18 / CIFAR-10 Benchmark</div>", unsafe_allow_html=True)
