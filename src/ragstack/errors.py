"""Exception hierarchy."""


class RagStackError(Exception):
    pass


class ConfigError(RagStackError):
    pass


class ProviderError(RagStackError):
    pass


class StoreError(RagStackError):
    pass


class ToolError(RagStackError):
    pass
