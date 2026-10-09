"""Feature extraction: MFDFA and band-power (PSD) features.

``mfdfa`` is the algorithm, ``surrogates`` and ``synthetic`` its test signals, and ``envelope``
the band envelopes it is also run on. ``scaling`` and ``scaling_report`` look at where the EEG
scales, without the group labels, to choose the fit ranges. ``extract`` runs MFDFA with
surrogates on every recording, ``psd`` computes the band powers, ``tables`` builds the long and
wide tables, and ``report`` writes their QC report.
"""
