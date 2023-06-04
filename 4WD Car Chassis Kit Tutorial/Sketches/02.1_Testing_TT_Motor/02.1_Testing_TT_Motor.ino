/**********************************************************************
  Product     : Cokoino 4WD Car chassis kit
  Auther      : www.cokoino.com
  Modification: 2022/07/01
**********************************************************************/

#include <AFMotor.h>
 AF_DCMotor motor1(1);//define motor1
 AF_DCMotor motor2(2);//define motor2
 AF_DCMotor motor3(3);//define motor3
 AF_DCMotor motor4(4);//define motor4

void setup() 
{
  //Set initial speed of the motor & stop
   motor1.setSpeed(200);//setup the speed of motor1
   motor2.setSpeed(200);//setup the speed of motor2
   motor3.setSpeed(200);//setup the speed of motor3
   motor4.setSpeed(200);//setup the speed of motor4
   
}

void loop() 
{
  motor1.run(FORWARD);// motor1 run FORWARD
  motor2.run(FORWARD);
  motor3.run(FORWARD);
  motor4.run(FORWARD);
  delay(3000);
  motor1.run(RELEASE);// motor1 stop run
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
  delay(3000);
  motor1.run(BACKWARD);//motor1 run BACKWARD
  motor2.run(BACKWARD);
  motor3.run(BACKWARD);
  motor4.run(BACKWARD);
  delay(3000);
  motor1.run(RELEASE);
  motor2.run(RELEASE);
  motor3.run(RELEASE);
  motor4.run(RELEASE);
   delay(3000);
}
