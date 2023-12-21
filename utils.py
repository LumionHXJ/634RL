import time

def time_counter(attribute_name):
    """A decorator that records the cumulative execution time of different 
    functions and stores it in a class attribute.

    ALERT! not allowing using cls_attr with name 'time'.

    Usage:
    ```
    @time_counter('time')
    def func(self, *args, **kwargs):
        pass
    ```
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            start_time = time.time()  # Start the timer
            result = func(self, *args, **kwargs)  # Call the function
            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time
            if hasattr(type(self), attribute_name):
                setattr(type(self), attribute_name, getattr(type(self), attribute_name) + elapsed_time)
            else:
                setattr(type(self), attribute_name, elapsed_time)
            return result
        return wrapper
    return decorator

def time_stats(*classes):
    """A decorator to clear and print time-related class variables of the 
    specified classes. It can be applied to any method of a class.

    Args:
        *cls: A variable number of class objects whose time-related class 
        attributes will be recounted and printed. (including self's class)
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # clear cache of time counter
            nonlocal classes
            
            for cl in classes + (type(self), ):
                to_modify = [name for name in vars(cl).keys() if 'time' in name]
                for name in to_modify:
                    setattr(cl, name, 0)
            
            result = func(self, *args, **kwargs)  # Call the function

            # output time used
            for cl in  classes + (type(self), ):
                for name, value in vars(cl).items():
                    if 'time' in name:
                        print(f"Device {self.device} {name}: {value}")
            return result
        return wrapper
    return decorator