import cv2
import serial
import keyboard
import time

# Arduino seri port bağlantısı (COMx yerine portunu yaz, Linux için /dev/ttyUSBx olabilir)
arduino = serial.Serial('COM10', 9600)
time.sleep(2)  # Bağlantı kurulana kadar bekle

# Kamera aç
cap = cv2.VideoCapture(0)

print("Kamera açık. Tuşlara basarak motorları kontrol edebilirsin.")

try:
    while True:
        ret, frame = cap.read()
        cv2.imshow('Kamera', frame)

        # Tuş kontrolleri
        if keyboard.is_pressed('w'):
            print("Plastik")
            arduino.write(b'w\n')
            time.sleep(0.3)

        elif keyboard.is_pressed('e'):
            print("Cam")
            arduino.write(b'e\n')
            time.sleep(0.3)

        elif keyboard.is_pressed('r'):
            print("Metal")
            arduino.write(b'r\n')
            time.sleep(0.3)

        elif keyboard.is_pressed('a'):
            print("Maksimum Açılma")
            arduino.write(b'a\n')
            time.sleep(0.3)

        elif keyboard.is_pressed('k'):
            print("Başlangıç Pozisyonu")
            arduino.write(b'k\n')
            time.sleep(0.3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()
arduino.close()
