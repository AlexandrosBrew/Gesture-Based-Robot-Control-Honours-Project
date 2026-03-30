import time

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        
        self.prev_error = 0
        self.integral = 0
        self.prev_time = time.time()

    def compute(self, current_value):
        curr_time = time.time()
        dt = curr_time - self.prev_time

        # Prevent division by zero on the first loop
        if dt <= 0.0:
            dt = 1e-4

        error = self.setpoint - current_value
        
        # Calculate P, I, and D terms
        proportional = self.kp * error
        self.integral += error * dt
        derivative = self.kd * ((error - self.prev_error) / dt)

        output = proportional + (self.ki * self.integral) + derivative

        # Save state for the next loop
        self.prev_error = error
        self.prev_time = curr_time

        return output