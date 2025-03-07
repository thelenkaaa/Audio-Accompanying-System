from LLM.config import ERRORS_SERVICE_UNAVAILABLE, ERRORS_TO_RETRY, ERRORS_TO_SKIP


def exponential_retry(func):
    """
    Retry decorator with exponential backoff for OpenAI API calls.

    This decorator adds a retry mechanism with increasing delays to OpenAI API calls.
    It's intended to handle temporary errors like rate limiting or network issues.

    :param func: callable - The function to decorate.
    :return: callable - The decorated function with retry mechanism.
    """
    exponential_base = 2
    initial_delay = 2
    max_delay = 600

    def wrapper(*args, **kwargs):
        """
        Execute the provided function with exponential backoff retries.

        :param args: tuple - Positional arguments for the function.
        :param kwargs: dict - Keyword arguments for the function.
        :return: object - Result of the function call.
        """
        delay = initial_delay

        while True:
            try:
                # Attempt to execute the provided function
                return func(*args, **kwargs)
            except ERRORS_SERVICE_UNAVAILABLE as e:
                raise e
            except ERRORS_TO_SKIP as e:
                print(f"[ERROR] Bad request, skipping processing... Error: {e}")
                return None
            except ERRORS_TO_RETRY as e:

                # Increase the delay time using an exponential factor
                delay *= exponential_base

                # If the delay exceeds the maximum, raise an exception
                if delay > max_delay:
                    print(f"[ERROR] Exceeded maximum retry time ({max_delay}). Error: {e}")
                    return None

                # Print an error message and wait before retrying
                print(f"[ERROR] Waiting {delay} seconds because of the error: {e}")
                time.sleep(delay)

    return wrapper