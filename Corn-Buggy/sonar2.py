import serial

if __name__== '__main__':
    ser=serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    ser.reset_input_buffer()
    
    while True:
        if ser.in_waiting>0:
            line=ser.read(ser.in_waiting)#.decode('utf-8').lstrip('R')
            print(line)
