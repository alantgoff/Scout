"""Multi-agent diligence engine — investment-decision depth on demand.

Two-tier scoring: the screening score (scout.score) keeps ranking the feed;
the Diligence Score built here (0–100 composite of 8 researched dimensions)
attaches to startups the user chooses to analyze. Never conflate the two.

Layout:
  schema.py     pydantic models + pure parse helpers (JSON-in-text idiom)
  rubrics.py    the 8 dimension system prompts + recon/cross-exam/memo prompts
  research.py   one Messages API call with web tools, pause_turn loop, usage ledger
  composite.py  pure weighted math -> 0-100 (mirrors score.score_breakdown)
  pipeline.py   run_analysis orchestrator (recon -> dims -> cross-exam -> memo -> KG)
  graph.py      knowledge-graph pure logic (persistence lives in store.py)
  priority.py   $0 "worth a deeper look" ranking from existing data
"""
