# frontend/components/results.py
import streamlit as st
import numpy as np


class ResultsDisplay:
    def __init__(self):
        self.modalities = ["T1", "T1ce", "T2", "FLAIR"]
        self.tumor_labels = {0: "Tumor Core", 1: "Whole Tumor", 2: "Enhancing Tumor"}
        self.colors = {
            0: [1, 0, 0],  # Red for Tumor Core
            1: [0, 1, 0],  # Green for Whole Tumor
            2: [0, 0, 1],  # Blue for Enhancing Tumor
        }

    def show(self, original_img, predictions):
        """
        Display the segmentation results alongside original image
        Args:
            original_img: shape (4, H, W, D) - original BraTS modalities
            predictions: shape (3, H, W, D) - predicted segmentation masks for each tumor type
        """
        # Select modality for background
        selected_modality = st.radio(
            "Select Background Modality",
            self.modalities,
            horizontal=True,
            key="result_modality",
        )
        modality_idx = self.modalities.index(selected_modality)

        # Get the selected modality data
        modality_data = original_img[modality_idx]

        # Select visualization slice
        slice_idx = st.slider(
            "Select Slice",
            0,
            modality_data.shape[2] - 1,
            modality_data.shape[2] // 2,
            key="result_slice",
        )

        # Create columns for display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Original {selected_modality}")
            original_slice = modality_data[:, :, slice_idx]
            st.image(self._normalize_slice(original_slice), use_container_width=True)

        # Add tumor region toggles and get visibility states
        st.subheader("Tumor Regions")
        cols = st.columns(len(self.tumor_labels))
        visible_regions = {}

        for col, (idx, name) in zip(cols, self.tumor_labels.items()):
            with col:
                st.markdown(f"**{name}**")
                color = st.color_picker(
                    "Color",
                    value=f"#{int(self.colors[idx][0]*255):02x}"
                    f"{int(self.colors[idx][1]*255):02x}"
                    f"{int(self.colors[idx][2]*255):02x}",
                    key=f"color_{idx}",
                    disabled=True,
                )
                # Add toggle for region visibility
                visible_regions[idx] = st.toggle(
                    "Show Region", value=True, key=f"toggle_{idx}"  # Default to visible
                )
                mask = (
                    tumor_slices[idx]
                    if "tumor_slices" in locals()
                    else predictions[idx, :, :, slice_idx]
                )
                st.metric("Volume (pixels)", value=int(np.sum(predictions[idx])))

        with col2:
            st.subheader("Segmentation Overlay")
            # Get segmentation slice for each tumor type
            tumor_slices = predictions[:, :, :, slice_idx]  # Shape: (3, H, W)
            # Create overlay with visibility control
            overlay = self._create_overlay(
                original_slice, tumor_slices, visible_regions
            )
            st.image(overlay, use_container_width=True)

    def _normalize_slice(self, slice_data):
        """Normalize slice data to 0-1 range"""
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_min == slice_max:
            return slice_data
        return (slice_data - slice_min) / (slice_max - slice_min)

    def _create_overlay(self, original_slice, tumor_slices, visible_regions):
        """Create colored overlay of segmentation on original image"""
        # Normalize original image
        orig_norm = self._normalize_slice(original_slice)

        # Create RGB image with original as background
        rgb = np.stack([orig_norm] * 3, axis=-1)

        # Add colored overlay for each tumor region
        for idx, color in self.colors.items():
            # Only show region if it's toggled on
            if visible_regions[idx]:
                mask = tumor_slices[idx] > 0
                if np.any(mask):  # Only process if region exists
                    # Create colored overlay with alpha blending
                    alpha = 0.3  # Transparency factor
                    for c in range(3):
                        rgb[mask, c] = (1 - alpha) * rgb[mask, c] + alpha * color[c]

        return rgb

    def _calculate_volume_stats(self, segmentation_mask):
        """Calculate volume statistics for each tumor region"""
        stats = {}
        for label, name in self.tumor_labels.items():
            volume = np.sum(segmentation_mask == label)
            stats[name] = volume
        return stats
