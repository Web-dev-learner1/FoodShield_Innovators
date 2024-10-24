import serial
import time

def init_serial(port='COM5', baudrate=9600):
    ser = serial.Serial(port=port, baudrate=baudrate, timeout=1)
    time.sleep(2)
    return ser

def send_command(ser, command):
    ser.write(command.encode())
    print(f"Sent: {command}")

def work(command):
    try:
        ser = init_serial()
        if command in ['0', '1']:
            send_command(ser, command)
        else:
            print("Invalid command. Please send '0' or '1'.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ser.close()
        print("Serial port closed")
