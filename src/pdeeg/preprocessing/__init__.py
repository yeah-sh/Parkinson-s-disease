"""Raw recordings to continuous cleaned recordings.

``pipeline`` crops, filters, re-references, removes ICA components and marks bad stretches;
``ica`` and ``segments`` hold those two steps; ``qc`` draws the figures and ``report`` writes
the QC report.
"""
