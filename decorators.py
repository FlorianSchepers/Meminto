import time


def log_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        with open("output/time_log.txt", "a") as time_log_file:
            time_log_file.write(f"{func.__name__!r}: {execution_time:.2f}s\n")
        print(f"Finished {func.__name__!r} in {execution_time:.2f}s")
        return output

    return wrapper
