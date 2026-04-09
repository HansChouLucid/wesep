try:
    from wesep.cli.extractor import load_model  # noqa
    from wesep.cli.extractor import load_model_local  # noqa
except Exception:  # pragma: no cover
    # Allow utility scripts to import wesep modules without pulling in
    # optional runtime/model dependencies such as wespeaker.
    load_model = None
    load_model_local = None
