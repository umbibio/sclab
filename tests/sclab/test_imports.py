def test_sclab_imports():
    from sclab import __version__

    assert __version__

    from sclab import SCLabDashboard

    assert SCLabDashboard

    from sclab.dataset.processor.step import ProcessorStepBase

    assert ProcessorStepBase
