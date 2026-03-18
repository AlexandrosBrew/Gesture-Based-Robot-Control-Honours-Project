#include <Servo.h>


// Wirsten in little-endian format: [Header (1 byte), Servo ID (1 byte), Position (2 bytes)]
// Gripper ID = 0, Wrist ID = 1, Elbow ID = 2, Base ID = 3, 
struct __attribute__((packed)) ServoCommand {
  uint8_t id;
  int16_t position; //Signed short for negative direction.
};

ServoCommand cmd;
const uint8_t HEADER = 255;
const int8_t wristAngle = 0;
const int8_t baseAngle = 0;
const int16_t gripperAngle = 0;
const int16_t elbowAngle = 0;

void setup() {
  Serial.begin(115200);
  //Set angles to 0.
}

void loop() {
  // We need at least 4 bytes now (1 header + 3 data)
  if (Serial.available() >= 4) {
    // Check if the first byte in the buffer is our header
    if (Serial.read() == HEADER) {
      // If yes, read the next 3 bytes directly into our struct
      Serial.readBytes((char*)&cmd, sizeof(cmd));

      switch(cmd.id){
        case 0: {
          //Gripper value 0-180
          Serial.println("Gripper Val: " + String(cmd.position));
          break;
        }

        case 1: {
          //Wrist value +10, -10 or 0 to stop
          Serial.println("Wrist Val: " + String(cmd.position));
          break;
        }

        case 2: {
          // Elbow value 0-180
          Serial.println("Elbow Val: " + String(cmd.position));
          break;
        }

        case 3: {
          // Base value +10, -10 or 0 to stop
          Serial.println("Base Val: " + String(cmd.position));
          break;
        }

        default: {
          //Continue
          break;
        }
      }
    } 
    // If the byte wasn't 255, the loop just continues, 
    // effectively "tossing" the bad byte until it finds a 255.
  }
  else{
  }
}

class packetTesting {
};