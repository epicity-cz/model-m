import os


def file_cache(filename):
    def decorator(func):
        def wraps(*args, **kwargs):
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    return f.read()
            data = func(*args, **kwargs)
            real = os.path.realpath(filename)
            real_dir = os.path.dirname(real)
            if not os.path.exists(real_dir):
                os.makedirs(real_dir)
            with open(real, 'wb') as out:
                out.write(data)
            return data

        return wraps

    return decorator
