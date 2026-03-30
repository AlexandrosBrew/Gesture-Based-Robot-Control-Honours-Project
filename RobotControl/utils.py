
def angle_to_microseconds(angle):
    """Helper to convert 0-180 degrees to 500-2500 microseconds."""
    return 500 + (angle / 180.0) * 2000

def check_limits(id, theta):
    """Check if the given angle is within the limits for the specified joint."""
    limits = {
        0: (0, 360),   # Base
        1: (-90, 180), # Shoulder
        2: (-150, 150),# Elbow
        3: (-180, 180) # Wrist
    }
    if id in limits:
        lo, hi = limits[id]
        return lo <= theta <= hi
    return False

def map_angle_to_pwm(angle, angle_min, angle_max):
    """
    Maps an angle to a PWM pulse width (500-2500).
    """
    pwm_min = 500
    pwm_max = 2500
    
    # Linear mapping formula
    pwm = ((angle - angle_min) * (pwm_max - pwm_min) / (angle_max - angle_min)) + pwm_min
    
    # Constrain the value to stay within 500-2500
    return max(pwm_min, min(pwm_max, int(pwm)))