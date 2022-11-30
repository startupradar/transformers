class StartupRadarAPIError(RuntimeError):
    pass


class NotFoundError(StartupRadarAPIError):
    pass


class ForbiddenError(StartupRadarAPIError):
    pass


class StartupRadarAPIWrapperError(RuntimeError):
    pass


class InvalidDomainError(StartupRadarAPIWrapperError):
    pass
