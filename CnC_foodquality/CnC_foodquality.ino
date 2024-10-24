int spoilt = 13;

void setup() {
  Serial.begin(9600);
  pinMode(spoilt, OUTPUT);
}

void loop() {
  if(Serial.available()> 0){
    char c = Serial.read();
    if(c == '1'){
      digitalWrite(spoilt, LOW);
    }
    else if(c == '0'){
      digitalWrite(spoilt, HIGH);
    }
  }

}
