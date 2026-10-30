"""A minimal stand-in for the rpy2 pieces used by the pseudobulk wrappers.

It lets the Python-side orchestration (aggregation, contrast naming, pct/num/tot
annotation, error handling) be tested without an R installation.
"""

from contextlib import contextmanager

import pandas as pd


class FakeR:
    def __init__(self, tt_columns, fail_on=None):
        self.calls: list[str] = []
        self.assigned: dict = {}
        self.tt_columns = tt_columns
        self.fail_on = fail_on  # substring of an R snippet that should error

    def __call__(self, code: str):
        self.calls.append(code)
        if self.fail_on is not None and self.fail_on in code:
            from rpy2.rinterface_lib.embedded import RRuntimeError

            raise RRuntimeError("simulated R failure")
        return "<r-object>"

    def assign(self, name, value):
        self.assigned[name] = value

    def fake_table(self) -> pd.DataFrame:
        genes = self.assigned["aggr_adata"].var_names
        data = {
            c: [0.1 * (i + 1) for i, _ in enumerate(genes)] for c in self.tt_columns
        }
        return pd.DataFrame(data, index=pd.Index(genes))

    def contrasts(self) -> list[str]:
        out = []
        for c in self.calls:
            if "makeContrasts(" in c:
                out.append(c.split("makeContrasts('")[1].split("'")[0])
        return out


def install_fake_r(monkeypatch, module, tt_columns, fail_on=None) -> FakeR:
    import rpy2.robjects as robjects
    import rpy2.robjects.conversion as conversion
    from rpy2.robjects import pandas2ri

    fake = FakeR(tt_columns, fail_on=fail_on)

    @contextmanager
    def _null_converter(_):
        yield

    monkeypatch.setattr(module, "_try_imports", lambda: None)
    monkeypatch.setattr(robjects, "r", fake)
    monkeypatch.setattr(conversion, "localconverter", _null_converter)
    monkeypatch.setattr(pandas2ri, "rpy2py", lambda _obj: fake.fake_table())
    return fake
