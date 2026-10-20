import numpy as np
from saleae.range_measurements import AnalogMeasurer

class RiseFallTime(AnalogMeasurer):
    supported_measurements = ["v_10p", "v_90p", "t_rise", "t_fall"]

    def __init__(self, requested_measurements):
        super().__init__(requested_measurements)
        self.samples = []
        self.sampling_period = None

        self.measure_v_10p = "v_10p" in requested_measurements
        self.measure_v_90p = "v_90p" in requested_measurements
        self.measure_t_rise = "t_rise" in requested_measurements
        self.measure_t_fall = "t_fall" in requested_measurements

    def process_data(self, data):
        self.samples.append(data.samples)
        if self.sampling_period is None and data.sample_count > 0:
            self.sampling_period = float((data.end_time - data.start_time) / data.sample_count)

    def measure(self):
        if not self.samples:
            return {}

        data = np.concatenate(self.samples)
        if len(data) < 2:
            return {}

        v_min = float(np.min(data))
        v_max = float(np.max(data))
        v_pp = v_max - v_min

        # Prevent calculations if signal is completely flat or noise-only (< 50mV)
        if v_pp < 0.05:
            return {}

        v_10p = v_min + (v_pp * 0.1)
        v_90p = v_min + (v_pp * 0.9)

        values = {}

        if self.measure_v_10p:
            values["v_10p"] = v_10p
        if self.measure_v_90p:
            values["v_90p"] = v_90p

        # Find crossings using numpy vector operations
        above_10 = data >= v_10p
        above_90 = data >= v_90p

        # Detect edge transitions (False -> True for rising, True -> False for falling)
        cross_10_rise = np.where(~above_10[:-1] & above_10[1:])[0]
        cross_90_rise = np.where(~above_90[:-1] & above_90[1:])[0]

        cross_90_fall = np.where(above_90[:-1] & ~above_90[1:])[0]
        cross_10_fall = np.where(above_10[:-1] & ~above_10[1:])[0]

        # Calculate Rise Time
        if self.measure_t_rise and len(cross_10_rise) > 0 and len(cross_90_rise) > 0:
            t_rises = []
            for idx_10 in cross_10_rise:
                # Find the first 90% crossing that occurs AFTER this 10% crossing
                valid_90 = cross_90_rise[cross_90_rise > idx_10]
                if len(valid_90) > 0:
                    idx_90 = valid_90[0]
                    t_rises.append((idx_90 - idx_10) * self.sampling_period)
            
            if len(t_rises) > 0:
                values["t_rise"] = float(np.mean(t_rises))

        # Calculate Fall Time
        if self.measure_t_fall and len(cross_90_fall) > 0 and len(cross_10_fall) > 0:
            t_falls = []
            for idx_90 in cross_90_fall:
                # Find the first 10% crossing that occurs AFTER this 90% crossing
                valid_10 = cross_10_fall[cross_10_fall > idx_90]
                if len(valid_10) > 0:
                    idx_10 = valid_10[0]
                    t_falls.append((idx_10 - idx_90) * self.sampling_period)
            
            if len(t_falls) > 0:
                values["t_fall"] = float(np.mean(t_falls))

        return values
