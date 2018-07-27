import threading


class AtomicTimeVar:
    def __init__(self, initial_val=None, initial_time=None):
        if initial_time is None:
            initial_time = 0

        self.latest_time = initial_time

        self.val = initial_val

        self.lock = threading.Lock()

    def set_val(self, new_val, timestamp):
        # if not None and (newer or more than 1 minute older (in case of uint32 wrapping)
        # 4293976030
        #    \/
        # 1009022
        if new_val is not None and (self.latest_time < timestamp or self.latest_time - timestamp > 60000000):
            # if self.latest_time - timestamp > 60000000:
            #     print("WRAPPED")
            self.lock.acquire()

            self.val = new_val
            self.latest_time = timestamp

            self.lock.release()

            return True
        else:
            return False

    def get_val(self):
        self.lock.acquire()
        val = self.val
        self.lock.release()
        return val


