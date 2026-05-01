import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class IKResult:
    """Result of an inverse kinematics calculation for a 3DoF RRR arm."""
    theta1: float      # Base rotation
    theta2: float      # Shoulder angle
    theta3: float      # Elbow angle
    elbow_up: bool

    def __repr__(self):
        return (
            f"IKResult(\n"
            f"  theta1 = {self.theta1:.3f}° (Base)\n"
            f"  theta2 = {self.theta2:.3f}° (Shoulder)\n"
            f"  theta3 = {self.theta3:.3f}° (Elbow)\n"
            f"  elbow  = {'up' if self.elbow_up else 'down'}\n"
            f")"
        )

class RobotKinematics:
    """
    Inverse kinematics solver for a 3DoF RRR robot arm with:
      - theta1: rotating base
      - theta2: shoulder
      - theta3: elbow

    Geometry:
      L1: base height (vertical offset from ground to shoulder joint)
      L2: shoulder to elbow
      L3: elbow to end effector
    """

    DEFAULT_L1 = 56.0    # Base height
    DEFAULT_L2 = 120.0   # Shoulder to elbow
    DEFAULT_L3 = 117.0   # Elbow to end effector

    def __init__(self, L1=DEFAULT_L1, L2=DEFAULT_L2, L3=DEFAULT_L3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def inverse(self, px: float, py: float, pz: float, elbow_up: bool = True) -> Optional[IKResult]:
        # 1. Base rotation
        t1 = math.atan2(py, px)

        # 2. Convert target into planar coordinates for shoulder/elbow
        r = math.sqrt(px**2 + py**2)   # horizontal distance from base axis
        z = pz - self.L1               # vertical distance from shoulder joint

        # 3. Law of cosines for elbow
        D = (r**2 + z**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)

        if abs(D) > 1.0:
            return None  # Target is unreachable

        # Elbow configuration
        if elbow_up:
            t3 = math.atan2(math.sqrt(max(0.0, 1.0 - D**2)), D)
        else:
            t3 = math.atan2(-math.sqrt(max(0.0, 1.0 - D**2)), D)

        # 4. Shoulder angle
        t2 = math.atan2(z, r) - math.atan2(self.L3 * math.sin(t3), self.L2 + self.L3 * math.cos(t3))

        # Convert to degrees
        theta1 = math.degrees(t1) % 360.0
        theta2 = math.degrees(t2)
        theta3 = math.degrees(t3)

        return IKResult(theta1, theta2, theta3, elbow_up)

if __name__ == "__main__":
    robot = RobotKinematics()

    # Example target: 150 mm forward, 50 mm right, 100 mm high
    result = robot.inverse(150, 50, 100, elbow_up=True)

    if result is None:
        print("Target is out of reach.")
    else:
        print(result)