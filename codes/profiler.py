import time

class Profiler:
    def __init__(self, interval = 1):
        self.start_time = None
        self.interval_count = 0
        self.interval = interval

    def start(self):
        self.start_time = time.time()

    def get_elapsed_time(self):
        if self.start_time is None:
            raise ValueError("Profiler has not been started.")
        elapsed_time = time.time() - self.start_time
        return elapsed_time
    
    def past_interval(self):
        if self.start_time is None:
            raise ValueError("Profiler has not been started.")
        
        elapsed = self.get_elapsed_time() - self.interval_count * self.interval
        if elapsed >= self.interval:
            self.interval_count += 1
            return True
        return False