import serial 
import time 
import struct

class Comm:
    def __init__(self):
        """Initialize the serial connection to the Arduino.

        This attempts to open the configured serial port and waits briefly for the Arduino to reset.

        Args:
            None
        """
        self.arduino = None
        try:
            self.arduino = serial.Serial(port='/dev/cu.usbmodem1301', baudrate=115200, timeout=.1)
            time.sleep(2) # Wait for arduino to reset after connection 
        except Exception as e:
            print("Connection Failed: ", e)

    def send_servo_command(self, servo_id, position_us):
        """Send a servo position command to the Arduino using a simple packet format. This tells a specific servo to move to the requested pulse width position.
        Wirsten in little-endian format: [Header (1 byte), Servo ID (1 byte), Position (2 bytes)]
        Gripper ID = 0, Wrist ID = 1, Elbow ID = 2, Base ID = 3, 
        Args:
            servo_id: The numeric identifier of the target servo.
            position_us: The desired servo position expressed in microseconds.
        """
        header = 255 # Adding header byte for reliability. Waits for the unique header before reading the rest of the packet.
        packet = struct.pack('<BBh', header, servo_id, position_us)
        self.arduino.write(packet)

if __name__ == "__main__":
    comm = Comm()
    while True:
        comm.send_servo_command(0, 1500)  # Example: Move gripper to neutral position (1500us)
        print("Sent command to move gripper to 1500us")
        time.sleep(1)  # Wait for a second before sending the next command
        