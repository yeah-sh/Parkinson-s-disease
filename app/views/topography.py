"""Scalp Topography: one feature over the scalp, large, with the channels ranked."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from neurolab import charts, engine, figures
from neurolab import components as ui
from neurolab.glossary import TERMS
from pdeeg import demo
from pdeeg.features.psd import feature_name
from pdeeg.features.scaling import BROADBAND
from pdeeg.features.tables import add_regions

ui.page_header("topography")
found = ui.need_analysis("topography")
if found is None:
    st.stop()
config = engine.settings()
mfdfa = config.mfdfa
index = ui.pick_recording(found, "topography")
described = found.described[index]
channels = described.series.channels
MFDFA, POWER = "MFDFA feature", "Band power"
SIGNED = ("delta_h", "asymmetry")

with st.container(key="nl-card-controls"):
    source = st.segmented_control(
        "Source",
        [MFDFA, POWER],
        default=MFDFA,
        required=True,
        key="nl-topo-source",
        persist_state="session",
    )
    if source == MFDFA:
        a, b, c = st.columns(3, gap="small")
        name = a.selectbox(
            "Series",
            [BROADBAND, *mfdfa.envelope.bands],
            format_func=lambda n: (
                "Broadband EEG" if n == BROADBAND else f"{n.capitalize()} envelope"
            ),
            key="nl-topo-series",
            persist_state="session",
        )
        ranges = demo.ranges_in_seconds(name, config)
        if st.session_state.get("nl-topo-range") not in ranges:
            st.session_state.pop("nl-topo-range", None)
        fit_range = b.selectbox(
            "Fit range",
            list(ranges),
            format_func=lambda r: f"{r} · {ranges[r][0]:g}–{ranges[r][1]:g} s",
            key="nl-topo-range",
            persist_state="session",
        )
        section = mfdfa.broadband if name == BROADBAND else mfdfa.envelope
        used = set(section.wide_features[fit_range])
        feature = c.selectbox(
            "Feature",
            list(mfdfa.features),
            format_func=lambda f: f + ("" if f in used else " (not a model feature here)"),
            key="nl-topo-feature",
            persist_state="session",
            help="The definition of the feature chosen is under the map.",
        )
        table = engine.channel_table(engine.key(), index, name, fit_range)
        values = table.set_index("channel").reindex(channels)[feature].to_numpy(dtype=float)
        signed = feature in SIGNED
        title = f"{feature} · {'broadband EEG' if name == BROADBAND else name + ' envelope'}, "
        title += f"{fit_range} range"
        unit = feature
        definition = TERMS.get(feature, "")
    else:
        a, b = st.columns([2, 1], gap="small")
        band = a.selectbox(
            "Band",
            list(config.psd.bands),
            format_func=lambda n: f"{n} ({config.psd.bands[n][0]:g}–{config.psd.bands[n][1]:g} Hz)",
            key="nl-topo-band",
            persist_state="session",
        )
        power = described.band_power
        clean = power[(power["band"] == band) & (power["segments"] == "clean")]
        values = clean.set_index("channel").reindex(channels)["value"].to_numpy(dtype=float)
        signed = False
        unit = feature_name(config.psd).replace("_", " ")
        title = f"{band} band · {unit}"
        b.html(
            f'<div class="nl-muted" style="font-size:0.8rem;padding-top:1.9rem">{unit}, '
            f"{config.psd.fmin:g}–{config.psd.fmax:g} Hz total, bad windows left out</div>"
        )
        feature = f"{band}_{feature_name(config.psd)}"
        definition = TERMS["Band power"]

finite = values[np.isfinite(values)]
left, right = st.columns([3, 2], gap="medium")
with left, st.container(key="nl-card-map"):
    st.markdown(f"**{ui.md(title)}**")
    st.pyplot(
        figures.scalp_maps(
            [figures.Map(values, "", signed)],
            channels,
            config.data.dataset.montage,
            names=True,
            unit=unit,
            panel=5.2,
        ),
        transparent=True,
        width="stretch",
    )
    missing = len(values) - len(finite)
    st.caption(
        "Interpolated between the sensors (MNE); every dot is a channel, labelled. "
        + ("Centred on 0: rose below, cyan above. " if signed else "Dim low, bright high. ")
        + (f"{missing} channel(s) without a value are left off. " if missing else "")
        + definition
    )
with right, st.container(key="nl-card-ranking"):
    if finite.size:
        ui.tiles(
            [
                {"label": "Lowest", "value": f"{finite.min():.3f}",
                 "sub": channels[int(np.nanargmin(values))]},
                {"label": "Median", "value": f"{np.median(finite):.3f}",
                 "sub": f"{finite.size} channels"},
                {"label": "Highest", "value": f"{finite.max():.3f}",
                 "sub": channels[int(np.nanargmax(values))]},
            ]
        )  # fmt: skip
    st.space("small")
    st.caption("Channels from highest to lowest; hover a bar for its value.")
    frame = pd.DataFrame({"channel": channels, "value": values})
    st.altair_chart(charts.ranking(frame, unit, signed), theme=None, width="stretch")

regional = add_regions(frame.assign(feature=feature), config.data.regions, ["feature"])
regional = regional[regional["level"] == "region"][["name", "value"]]
regional["channels"] = [", ".join(config.data.regions[r]) for r in regional["name"]]
with st.expander("Region means", icon=":material/grid_view:"):
    st.caption(
        "The mean of each region's channel values, NaN left out, formed by the pipeline's own "
        "function (pdeeg.features.tables.add_regions): the level of the features the models "
        f"use ({config.model.level})."
    )
    st.dataframe(
        regional.rename(columns={"name": "Region", "value": "Mean", "channels": "Channels"}),
        hide_index=True,
        width="stretch",
        column_config={"Mean": st.column_config.NumberColumn(format="%.4f")},
    )
