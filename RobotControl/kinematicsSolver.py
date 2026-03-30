import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class IKResult:
    """Result of an inverse kinematics calculation."""
    theta1: float      # Base rotation
    theta2: float      # Shoulder angle
    theta3: float      # Elbow angle
    theta4: float      # Manipulator pitch angle
    elbow_up: bool     
    in_range: bool     
    out_of_range_joints: list

    def __repr__(self):
        status = "in range" if self.in_range else f"OUT OF RANGE ({', '.join(self.out_of_range_joints)})"
        return (
            f"IKResult(\n"
            f"  theta1 = {self.theta1:.3f}° (Base)\n"
            f"  theta2 = {self.theta2:.3f}° (Shoulder)\n"
            f"  theta3 = {self.theta3:.3f}° (Elbow)\n"
            f"  theta4 = {self.theta4:.3f}° (Manipulator)\n"
            f"  elbow  = {'up' if self.elbow_up else 'down'}\n"
            f"  status = {status}\n"
            f")"
        )

class RobotKinematics:
    """
    Inverse kinematics solver for your 4-DOF RRR robot.
    
    L1: Base to Shoulder (56mm)
    L2: Shoulder to Elbow (120mm)
    L3: Elbow to Manipulator Motor (117mm)
    L4: Manipulator Motor to Tip (Set to 0 if targeting the motor center)
    """

    # Updated link lengths per your specifications
    DEFAULT_L1 = 56.0   
    DEFAULT_L2 = 120.0  
    DEFAULT_L3 = 117.0  
    DEFAULT_L4 = 105.0    # Change this if your gripper/tool has length

    # Joint limits (degrees) - Adjust these based on your specific servos/motors
    LIMITS = {
        "theta1": (0.0,   360.0),
        "theta2": (-90.0, 180.0),
        "theta3": (-150.0, 150.0),
        "theta4": (-180.0, 180.0),
    }

    def __init__(self, L1=DEFAULT_L1, L2=DEFAULT_L2, L3=DEFAULT_L3, L4=DEFAULT_L4):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4

    def inverse(self, px, py, pz, phi, elbow_up=True) -> Optional[IKResult]:
        phi_r = math.radians(phi)

        # 1. Base rotation
        t1 = math.atan2(py, px)

        # 2. Project to 2D plane and find Wrist Center
        r = math.sqrt(px**2 + py**2)
        # We find the position of the motor (joint 4)
        wx = r - self.L4 * math.cos(phi_r)
        wz = pz - self.L1 - self.L4 * math.sin(phi_r)

        # 3. Solve for Elbow (theta3) using Law of Cosines
        D = (wx**2 + wz**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)

        if abs(D) > 1.0:
            return None # Target out of reach

        sgn = 1.0 if elbow_up else -1.0
        t3 = math.atan2(sgn * math.sqrt(max(0.0, 1.0 - D**2)), D)
        
        # 4. Solve for Shoulder (theta2)
        t2 = math.atan2(wz, wx) - math.atan2(self.L3 * math.sin(t3), self.L2 + self.L3 * math.cos(t3))

        # 5. Solve for Manipulator Pitch (theta4)
        t4 = phi_r - t2 - t3

        # Convert to degrees and normalize
        theta1 = math.degrees(t1) % 360.0
        theta2 = math.degrees(t2)
        theta3 = math.degrees(t3)
        theta4 = math.degrees(t4)

        in_range, violations = self._check_limits(theta1, theta2, theta3, theta4)

        return IKResult(theta1, theta2, theta3, theta4, elbow_up, in_range, violations)

    def _check_limits(self, t1, t2, t3, t4):
        violations = []
        vals = {"theta1": t1, "theta2": t2, "theta3": t3, "theta4": t4}
        for name, val in vals.items():
            lo, hi = self.LIMITS[name]
            if not (lo <= val <= hi):
                violations.append(f"{name}={val:.2f}°")
        return len(violations) == 0, violations

if __name__ == "__main__":
    robot = RobotKinematics()
    # Example target: 150mm forward, 50mm right, 100mm high, 0 degree pitch
    result = robot.inverse(150, 50, 100, 0)
    print(result)