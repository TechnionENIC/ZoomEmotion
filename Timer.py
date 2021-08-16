import time


class Timer:
    total_time = 0
    last_measured_time = None
    state = "not start"
    time_after_stop = None

    def start(self):
        if self.state != "not start":
            raise Exception()
        self.state = "start"
        self.last_measured_time = time.time()

    def start_with_pause(self):
        if self.state != "not start":
            raise Exception()
        self.state = "pause"

    def pause(self):
        if self.state != "start":
            raise Exception()
        self.state = "pause"
        self.total_time += time.time() - self.last_measured_time

    def continue_(self):
        if self.state != "pause":
            raise Exception()
        self.state = "start"
        self.last_measured_time = time.time()

    def stop(self):
        if self.state != "start" and self.state != "pause":
            print("yoy stoped me already")
        self.state = "not start"
        self.time_after_stop = time.time() - self.last_measured_time
        self.last_measured_time = None
        self.total_time = 0

    def get_total_time(self):
        if self.time_after_stop is None:
            raise Exception()
        return self.time_after_stop

    def reset(self):
        self.stop()
        self.start_with_pause()