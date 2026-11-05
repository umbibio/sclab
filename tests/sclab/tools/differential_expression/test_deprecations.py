import pytest

from sclab.tools.differential_expression import (
    _pseudobulk_edger,
    _pseudobulk_limma,
    pseudobulk_edger,
    pseudobulk_limma,
)


@pytest.mark.parametrize(
    "func, mod, new",
    [
        (pseudobulk_limma, _pseudobulk_limma, "pseudobulk_pylimma"),
        (pseudobulk_edger, _pseudobulk_edger, "pseudobulk_edgepython"),
    ],
)
def test_r_backends_are_deprecated(counts_adata, monkeypatch, func, mod, new):
    class Stop(Exception):
        pass

    def stop():
        raise Stop

    monkeypatch.setattr(mod, "_try_imports", stop)
    with pytest.warns(FutureWarning, match=new), pytest.raises(Stop):
        func(counts_adata, "cond")
    assert ".. deprecated::" in func.__doc__
