from __future__ import annotations

import pytest

from course_project.scripts.build_pareto import validate_manifest_sources


def test_pareto_does_not_mix_parameter_sources():
    with pytest.raises(ValueError, match="source_id"):
        validate_manifest_sources({"sources": [{"source_id": "same", "name": "a"}, {"source_id": "same", "name": "b"}]})
    validate_manifest_sources({"sources": [{"source_id": "fixed10", "name": "fixed"}, {"source_id": "eps02", "name": "cakl"}]})
