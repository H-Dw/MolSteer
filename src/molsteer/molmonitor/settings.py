"""One explicit owner and switch for optional live monitoring and graph review."""


def monitor_settings(config):
    settings = config.get('monitor')
    if settings is None or settings is False:
        return None
    if not isinstance(settings, dict):
        raise ValueError('monitor must be an object or false')
    enabled = settings.get('enabled', True)
    if type(enabled) is not bool:
        raise ValueError('monitor.enabled must be boolean')
    return settings if enabled else None


def graph_review_settings(config):
    # The former sibling switch could run even with MolMonitor off. Fail with an
    # actionable migration message instead of silently making paid API calls.
    if config.get('graph_review'):
        raise ValueError('Move graph_review to monitor.graph_review and explicitly set enabled=true')
    monitor = monitor_settings(config)
    settings = monitor.get('graph_review') if monitor else None
    if settings is None or settings is False:
        return None
    if not isinstance(settings, dict):
        raise ValueError('monitor.graph_review must be an object or false')
    enabled = settings.get('enabled', False)
    if type(enabled) is not bool:
        raise ValueError('monitor.graph_review.enabled must be boolean')
    return settings if enabled else None
