OPENAI_API_KEY = "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"

# Errors for which the operation should be retried, as they may resolve on their own
ERRORS_TO_RETRY = (
    openai.Timeout,
    openai.RateLimitError,
    openai.InternalServerError
)
# Errors indicating issues with the request that should be skipped and not retried
ERRORS_TO_SKIP = (
    openai.BadRequestError,
    openai.UnprocessableEntityError
)
# Errors related to service availability or authentication issues that require further investigation
ERRORS_SERVICE_UNAVAILABLE = (
    openai.APIConnectionError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError
)