import pathlib
import time


def log_time(func):
    def wrapper(*args, **kwargs):
        time_log_file_path = (
            pathlib.Path(__file__).parent.resolve() / ".." / "output" / "time_log.txt"
        )
        if not time_log_file_path.exists():
            time_log_file_path.touch()

        start_time = time.time()
        output = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        end_timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(end_time))

        with open(time_log_file_path, "a") as time_log_file:
            time_log_file.write(
                f"[{end_timestamp}] Finished {func.__name__!r} in {execution_time:.2f}s\n"
            )
        print(f"[{end_timestamp}] Finished {func.__name__!r} in {execution_time:.2f}s")
        return output

    return wrapper
