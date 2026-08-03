"""The deal-flow status vocabulary — one layer below the UI.

Lived in ui.py while the Streamlit script was the only writer; now the
worker, notifications, insights, and CRM sync all read statuses, so the
vocabulary must not require importing Streamlit. Pure data, no I/O.
"""

from __future__ import annotations

STATUS_LABELS = {
    "new": "New",
    "longlisted": "Longlisted",
    "shortlisted": "Shortlisted",
    "contacted": "Contacted",
    "meeting": "Meeting",
    "diligence": "Diligence",
    "won": "Allocated",
    "passed": "Passed",
}
LABEL_TO_STATUS = {v: k for k, v in STATUS_LABELS.items()}
WIN_STAGES = ["shortlisted", "contacted", "meeting", "diligence", "won"]
# The triage funnel: feed → longlist → shortlist (→ stages to allocation).
FUNNEL_STAGES = ["longlisted", *WIN_STAGES]
# Any of these counts as a positive ("shortlisted or further") decision —
# the contrast class for triage insights and imported votes.
POSITIVE_STATUSES = {"longlisted", *WIN_STAGES}
