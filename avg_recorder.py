import numpy as np


class AvgRecorder:
    def __init__(self, num_frames):
        self.summed_frame = None
        self.cur_frame = 0
        self.max_frame = num_frames
        self.recording = False
        self.avg = None
        self.initialized = False

    @staticmethod
    def get_size(max_val):
        if max_val < 2 ** 8:
            return np.uint8
        elif max_val < 2 ** 16:
            return np.uint16
        elif max_val < 2 ** 32:
            return np.uint32
        elif max_val < 2 ** 64:
            return np.uint64
        elif max_val < 2 ** 128:
            return np.uint128
        else:
            raise ValueError("Val greater than 2**128: " + str(max_val))

    def initialize(self, shape, max_unq_val=256):
        self.summed_frame = np.zeros(shape, AvgRecorder.get_size(self.max_frame * max_unq_val))
        self.initialized = True

    def is_recording(self):
        return self.recording

    def begin_record(self):
        if not self.initialized:
            raise ValueError("Must be initialized first.")
        print("Beginning record...")
        self.recording = True
        # Reset
        self.summed_frame.fill(0)
        self.avg = None

    def record(self, frame):
        if not self.initialized:
            raise ValueError("Must be initialized first.")
        if self.is_recording():
            if self.cur_frame+1 > self.max_frame:
                print("Ending recording...")
                self.cur_frame = 0
                self.recording = False
                # Average and move 10 units forward for safety (gives some leeway)
                self.avg = np.subtract(np.divide(self.summed_frame, self.max_frame), 10).astype(np.uint8)
            else:
                self.summed_frame = np.add(frame, self.summed_frame)
                self.cur_frame += 1
        else:
            # Skip.
            pass

    def get_avg(self):
        if self.is_recording():
            return None
        else:
            return self.avg

