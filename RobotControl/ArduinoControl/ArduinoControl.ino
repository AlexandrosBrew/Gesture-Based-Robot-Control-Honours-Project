#include <Servo.h>

#define HEADER 0xFF

struct ServoCommand {
  uint8_t id;
  int16_t position;
};

ServoCommand cmd;

enum ParseState {
  WAIT_HEADER,
  READ_ID,
  READ_POS_L,
  READ_POS_H,
  READ_CHECKSUM
};

ParseState state = WAIT_HEADER;

uint8_t id;
uint8_t pos_l, pos_h;
uint8_t checksum;

uint16_t currGripperPosition = 0;
uint16_t currWristPosition = 0;
uint16_t currElbowPosition = 0;
uint16_t currBasePosition = 0;

int GripperPin = 5; 
int WristPin = 3;   
int ElbowPin = 2;   
int BasePin = 6;    
Servo Servos[4];
int ServoPins[4] = {GripperPin, WristPin, ElbowPin, BasePin};

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    Servos[i].attach(ServoPins[i]);
  }
  //Set initial positions
  Servos[0].writeMicroseconds(500); // Gripper (Closed)
  Servos[1].writeMicroseconds(1500); // Wrist (up)
  Servos[2].writeMicroseconds(1000); // Elbow (up)
  Servos[3].writeMicroseconds(1500); // Base (up)
}

void loop() {
  while (Serial.available()) {
    uint8_t byte_in = Serial.read();
    switch (state) {
      case WAIT_HEADER:
        if (byte_in == HEADER) {
          checksum = byte_in;
          state = READ_ID;
        }
        break;

      case READ_ID:
        id = byte_in;
        checksum ^= byte_in;
        state = READ_POS_L;
        break;

      case READ_POS_L:
        pos_l = byte_in;
        checksum ^= byte_in;
        state = READ_POS_H;
        break;

      case READ_POS_H:
        pos_h = byte_in;
        checksum ^= byte_in;
        state = READ_CHECKSUM;
        break;

      case READ_CHECKSUM:
        if (checksum == byte_in) {
          int16_t position = (int16_t)(pos_l | (pos_h << 8));
          // Validation layer
          if (id <= 3) {
            if (position >= 500 && position <= 2500) { // servo-safe range
              processCommand(id, position);
            } else {
              Serial.println("Invalid position");
            }
          } else {
            Serial.println("Invalid ID");
          }
        } else {
          Serial.println("Checksum fail");
        }

        state = WAIT_HEADER;
        break;
    }
  }
}

void processCommand(uint8_t id, int16_t position) {
  switch (id) {
    case 0:
      Serial.println("Gripper: " + String(position));
      // Map to servo range (500-1300)
      position = map(position, 500, 2500, 500, 1300);
      moveServo(id, position, currGripperPosition);
      currGripperPosition = position;
      break;
    case 1:
      Serial.println("Wrist: " + String(position));
      moveServo(id, position, currWristPosition);
      currWristPosition = position;
      break;
    case 2:
      Serial.println("Elbow: " + String(position));
      position = map(position, 500, 2500, 1000, 1700); // Elbow range
      moveServo(id, position, currElbowPosition);
      currElbowPosition = position;
      break;
    case 3:
      Serial.println("Base: " + String(position));
      // Base rotates full 360, so we can map 500-2500 to 0-360 degrees
      position = map(position, 500, 2500, 500, 2500);
      moveServo(id, position, currBasePosition);
      currBasePosition = position;
      break;
  }
}

void moveServo(uint8_t id, int16_t position, uint16_t currentPosition) {
  if (currentPosition != position) {
    Serial.println("Moving servo " + String(id) + " to position " + String(position));
    Servos[id].writeMicroseconds(position);
  }
}