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

int GripperPin = 9; 
int WristPin = 10;   
int ElbowPin = 11;   
int BasePin = 12;    
Servo Servos[4];
int ServoPins[4] = {GripperPin, WristPin, ElbowPin, BasePin};

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < 4; i++) {
    Servos[i].attach(ServoPins[i]);
  }
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
      currGripperPosition = position;
      break;
    case 1:
      Serial.println("Wrist: " + String(position));
      currWristPosition = position;
      break;
    case 2:
      Serial.println("Elbow: " + String(position));
      currElbowPosition = position;
      break;
    case 3:
      Serial.println("Base: " + String(position));
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