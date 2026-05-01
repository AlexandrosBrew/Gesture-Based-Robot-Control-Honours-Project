import serial
import time
import struct

class Comm:
    def __init__(self):
        """
        Initialize serial communication with the Arduino. Handles connection errors gracefully.
        """
        self.arduino = None
        try:
            self.arduino = serial.Serial(
                port='/dev/cu.usbmodem1101',
                baudrate=115200,
                timeout=0.1
            )
            time.sleep(2)
        except Exception as e:
            print("Connection Failed:", e)

    def _compute_checksum(self, header, servo_id, pos_l, pos_h):
        return header ^ servo_id ^ pos_l ^ pos_h

    def send_servo_command(self, servo_id, position_us):
        try:
            # Validation (important for robustness)
            if not (0 <= servo_id <= 3):
                raise ValueError("Invalid servo ID")

            if not (500 <= position_us <= 2500):
                raise ValueError("Position out of safe servo range")

            header = 0xFF

            # Pack without checksum first
            payload = struct.pack('<Bh', servo_id, position_us)
            pos_l = payload[1]
            pos_h = payload[2]

            checksum = self._compute_checksum(header, servo_id, pos_l, pos_h)

            packet = struct.pack('<BBhB', header, servo_id, position_us, checksum)

            self.arduino.write(packet)
        except Exception as e:
            print(f"Error sending {servo_id} command: {e}")

class packetTesting:
    def __init__(self, Comm):
        self.comm = Comm
        self.arduino = comm.arduino

    def send_corrupted_packet(self, servo_id, position_us):
        header = 0xFF
        packet = struct.pack('<BBh', header, servo_id, position_us)

        corrupted = bytearray(packet)
        corrupted[2] ^= 0xFF  # flip bits

        self.arduino.write(corrupted)
        print("-----Corrupted Packet Sent-----\n")
        print("Original Packet:", list(packet))
        print("Corrupted Packet:", list(corrupted))
        print("--------------------------------\n")

    def send_random_noise(self, length=10):
        import random
        noise = bytearray([random.randint(0, 255) for _ in range(length)])
        self.arduino.write(noise)
        print("-----Random Noise Sent-----\n")
        print("Noise Bytes:", list(noise))
        print("--------------------------------\n")

    def send_fragmented(self, servo_id, position_us):
        header = 0xFF
        payload = struct.pack('<Bh', servo_id, position_us)
        pos_l = payload[1]
        pos_h = payload[2]
        checksum = self.comm._compute_checksum(header, servo_id, pos_l, pos_h)

        packet = struct.pack('<BBhB', header, servo_id, position_us, checksum)

        for b in packet:
            self.arduino.write(bytes([b]))
            time.sleep(0.05)

        print("-----Fragmented Packet Sent-----\n")
        print("Packet Bytes:", list(packet))
        print("--------------------------------\n")

if __name__ == "__main__":
    comm = Comm()
    packetTesting = packetTesting(comm) 
    while True:
        inp = input("Enter testing command (or 'q' to quit): ")
        if inp == 'q':
            break
        elif inp.startswith("corrupt"):
            _, servo_id, pos = inp.split()
            packetTesting.send_corrupted_packet(int(servo_id), int(pos))
        elif inp.startswith("noise"):
            packetTesting.send_random_noise()
        elif inp.startswith("fragment"):
            _, servo_id, pos = inp.split()
            packetTesting.send_fragmented(int(servo_id), int(pos))
        elif inp.startswith("normal"):
            _, servo_id, pos = inp.split()
            comm.send_servo_command(int(servo_id), int(pos))
            print("-----Normal Packet Sent-----\n")
            print(f"Servo ID: {servo_id}, Position: {pos}us")
            print("--------------------------------\n")
        else:
            print("Unknown command. Use 'corrupt <servo_id> <position>', 'noise', or 'fragment <servo_id> <position>'.")