"""Trusted field locations for repair; never echo rejected values."""


class ContractValidationError(ValueError):
    def __init__(self, rule, path):
        super().__init__(rule)
        self.path = list(path)
