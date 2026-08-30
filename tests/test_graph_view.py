"""graph_view — the canvas builders. Pure, so the properties that matter are
directly testable: the filters shape the node set, and adversarial text from
the web can never break out of the page's data literal."""

from __future__ import annotations

from scout.graph_view import NODE_STYLE, TYPE_ORDER, graph_data, graph_page_html


def _edge(src_key, src_label, dst_key, dst_label, rel="invested_in",
          src_type="investor", dst_type="company", evidence=""):
    return {"src_type": src_type, "src_key": src_key, "src_label": src_label,
            "rel": rel, "dst_type": dst_type, "dst_key": dst_key,
            "dst_label": dst_label, "evidence": evidence}


EDGES = [
    _edge("sequoia", "Sequoia", "acme", "Acme"),
    _edge("sequoia", "Sequoia", "beta", "Beta"),
    _edge("lone", "Lone VC", "acme", "Acme"),
    _edge("karpathy", "@karpathy", "beta", "Beta", rel="follows",
          src_type="watcher"),
]


def test_cross_links_only_drops_single_company_connectors() -> None:
    nodes, links = graph_data(EDGES, cross_links_only=True)
    labels = {n["label"] for n in nodes}
    assert "Sequoia" in labels          # two companies — the pattern
    assert "Lone VC" not in labels      # restates one card's chips
    assert "Acme" in labels and "Beta" in labels  # companies always stay


def test_focus_keeps_the_two_hop_neighbourhood() -> None:
    far = _edge("othervc", "Other VC", "gamma", "Gamma")
    nodes, _ = graph_data(EDGES + [far], focus_key="acme")
    labels = {n["label"] for n in nodes}
    assert "Acme" in labels and "Sequoia" in labels and "Beta" in labels
    assert "Gamma" not in labels        # three hops out


def test_rel_filter_and_degree_counting() -> None:
    nodes, links = graph_data(EDGES, rels={"invested_in"})
    assert all(l["rel"] == "invested_in" for l in links)
    sequoia = next(n for n in nodes if n["label"] == "Sequoia")
    assert sequoia["degree"] == 2


def test_untrusted_labels_cannot_break_out_of_the_script() -> None:
    """Labels and evidence come from bios and websites. The classic breakout
    is a label containing "</script>" — it must never appear unescaped."""
    evil = _edge("evil", "</script><script>alert(1)</script>", "acme", "Acme",
                 evidence="</script><img src=x onerror=alert(2)>")
    nodes, links = graph_data([evil])
    html = graph_page_html(nodes, links)
    assert "</script><script>alert" not in html
    assert "</script><img" not in html


def test_every_node_type_has_a_validated_style() -> None:
    """The palette is validated as a set — a type added without a slot would
    silently render in someone else's color."""
    assert set(TYPE_ORDER) == set(NODE_STYLE)
    for style in NODE_STYLE.values():
        assert style["color"].startswith("#") and style["shape"]
    html = graph_page_html(*graph_data(EDGES))
    for style in NODE_STYLE.values():
        assert style["color"] in html   # legend + payload carry every slot
