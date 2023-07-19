#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_GFX.h>
#include <SPI.h>



#define OLED_ADDR   -1
#define OLED_SDA    4
#define OLED_SCL    5
Adafruit_SSD1306 display(128, 64, &Wire, OLED_ADDR);

// define the analog input pin
#define ANALOG_PIN  A0



// WiFi credentials
const char* ssid = "MDS";
const char* password = "12345678";

// MQTT broker information
const char* mqtt_server = "192.168.97.70";
const int mqtt_port = 1883;
//const char* mqtt_user = "your_MQTT_username";
//const char* mqtt_password = "your_MQTT_password";
const char* mqtt_topic = "m";

// Pin number for the pushbutton
const int buttonPin = 0;

// MQTT client
WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

// Time variables for sending analog read values
unsigned long lastMsg = 0;
const unsigned long interval = 2; // 200ms interval
unsigned long startTime = 0;

void setup() {

  Serial.begin(9600);
  Wire.begin(OLED_SDA, OLED_SCL);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  
  pinMode(buttonPin, INPUT_PULLUP);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");

  // Connect to MQTT broker
  mqttClient.setServer(mqtt_server, mqtt_port);
  mqttClient.setCallback(callback);
  while (!mqttClient.connected()) {
    Serial.println("Connecting to MQTT broker");
    if (mqttClient.connect("NodeMCU")) {
      Serial.println("Connected to MQTT broker");
    } else {
      Serial.print("Failed, rc=");
      Serial.println(mqttClient.state());
      delay(5000);
    }
  }
}

void loop() {

  int val = analogRead(ANALOG_PIN);
  // print the value to the serial plotter
  Serial.println(val);

  // display the value on the OLED display
  static int data[128] = {0};
  static int index = 0;
  data[index] = val;
  index = (index + 1) % 128;

  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.setTextColor(WHITE);
  display.println("ECG Signal:");
  display.drawLine(0, 11, 128, 11, WHITE);
  display.setCursor(0, 11);
  for (int i = 0; i < 128; i++) {
    int y = map(data[(index + i) % 128], 0, 1023, 63, 11);
    display.drawPixel(i, y, WHITE);
  }
  display.display();


  
  
  
  // Check for button press
  if (digitalRead(buttonPin) == LOW) {
    Serial.println("Button pressed");
//    for (int i = 0; i < 500; i++) { // Send 500 values in 4 seconds (125 values per second)
//      int analogVal = analogRead(A0);
//      mqttClient.publish(mqtt_topic, String(analogVal).c_str());
//      Serial.print("Analog value: ");
//      Serial.println(analogVal);
//      delay(8); // Delay 8 milliseconds between each message
//      mqttClient.loop();
//    }
//     mqttClient.publish(mqtt_topic, "a");

      long start_time = millis();
        int count = 0;
        
        while (millis() - start_time < 4000) { // read for 4 seconds
          int value = analogRead(A0);
          
          // publish to MQTT topic
          char message[10];
          sprintf(message, "%d", value);
          mqttClient.publish(mqtt_topic, message);
          
          count++;
          
          // delay to maintain 125 readings per second
          //delay(8);
        }
     mqttClient.publish(mqtt_topic, "a");
     char msg [10];
     sprintf(msg, "%d", count);
     mqttClient.publish("n", msg);
     Serial.println(count);

  }
}


void callback(char* topic, byte* payload, unsigned int length) {
  // Do something with incoming MQTT message if needed
}
