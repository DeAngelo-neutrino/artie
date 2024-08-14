############# Record the data here #################
import time
import serial
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


"""
Setup communications with the lakeshore model 218 temperature sensor readout.
configure the serial connections (the parameters differs on the device 
you are connecting to). Use COM for Windows, /dev/ttyUSB for Linux
Lakeshore baudrates can be 1200 or 9600, the Model 1700 for the
liquid level sensor has a baudrate of 115200.
"""
lakeshore_port = '/dev/ttyUSB1'
lakeshore_baudrate = 1200
model_1700_port = '/dev/ttyUSB0'
model_1700_baudrate = 115200

"""Set up serial interfaces"""
lakeshore = serial.Serial(
    port=lakeshore_port,
    baudrate=lakeshore_baudrate,
    parity=serial.PARITY_EVEN,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.SEVENBITS,
    timeout=1
)
model_1700 = serial.Serial(
    port=model_1700_port,
    baudrate=model_1700_baudrate,
    timeout=1
)

"""Get the device properties on these ports"""
lakeshore.flush()
if (lakeshore.isOpen()):
    print("Lakeshore port open")
model_1700.flush()
if (model_1700.isOpen()):
    print("model_1700 level sensor port open")

"""Set model 1700 to units of cm"""
model_1700.write('CONFigure:N2:UNIT CM'.encode(encoding = 'UTF-8'))

"""Set up output file"""
tnow = datetime.now()
t_tuple = datetime.timetuple(tnow)
timestamp = str(t_tuple.tm_mon) + "_" + \
    str(t_tuple.tm_mday) + "-" + str(t_tuple.tm_hour) + \
    "_" + str(t_tuple.tm_min) + "_" + str(t_tuple.tm_sec)
txtfname = "log_all_chan_" + timestamp + ".txt"
output_file = open(txtfname,"a")

"""Channel numbers for lakeshore"""
lakeshore_channel_num = [1,2,3,4,5,6,7,8]

lakeshore_channel_locations = [
    "Target Top", "Target Side", "Target Bottom","Front Plate", 
    "None", "None", "None", "None"
]
acquisition_time = 10 

"""Set up the start time for this run"""
TIME_0 = time.time()
output_file.write("Starting time for script = " + str(TIME_0) + "\n")

"""Output channel information"""
for x in range(len(lakeshore_channel_locations)) :
    line = "Location of channel " + str(lakeshore_channel_num[x]) + " is : " + lakeshore_channel_locations[x]
    output_file.write(line + "\n")
    print(line)

"""Construct header for output variables"""
top_line="Time (s)"
for x in range(len(lakeshore_channel_locations)) :
    top_line += ("," + lakeshore_channel_locations[x])
top_line += ',N2 Level'
output_file.write(top_line + "\n")
print(top_line)

"""Set up live plots"""
fig, ax1 = plt.subplots(figsize=(12,7))
ax2 = ax1.twinx()
ax1.set_xlabel("Time [s]")
ax2.set_ylabel("Liquid level [cm]")
ax1.set_ylabel("Temperature [K]")
ax1.set_title(f"ARTIE-II Sensor Readings {timestamp}")

t_data, n2_data, he_data = [], [], []
ch1_data, ch2_data, ch3_data, ch4_data = [], [], [], []
ch5_data, ch6_data, ch7_data, ch8_data = [], [], [], []

"""Main function to read from the lakeshore and model 1700"""
def read_all() :
    """
    Ask the Lakeshore for all channels at once as well as 
    a lakeshore_timestamp for the reading value.  The command
    for getting all output values is 'KRDG? 0 \r\n'
    """
    
    lakeshore_command = "KRDG? 0 \r\n"
    reading_time = round(time.time()-TIME_0, 2)
    lakeshore.write(lakeshore_command.encode(encoding = 'UTF-8'))
    readback = lakeshore.readline().decode()
    
    output_values = str(reading_time) + ","
    output_values += (readback.split())[0]
    
    """Ask the model_1700 level sensor for certain info"""
    command_n2 = 'MEASure:N2:LEVel?\n'
    model_1700.write(command_n2.encode(encoding = 'UTF-8'))
    n2_response = ''
    while n2_response == '':
        n2_response = model_1700.readline().decode().replace('\n','').replace('\r','')
    output_values += f',{n2_response}'
    
    print(output_values)
    output_file.write(output_values + "\n")
    
    """Clear the buffer to update output_file in real time"""
    output_file.flush()
    
    """Update live plot"""

    t_data.append(reading_time)
    lakeshore_values = readback.split(',')
    
    ch1_data.append(float(lakeshore_values[0]))
    ch2_data.append(float(lakeshore_values[1]))
    ch3_data.append(float(lakeshore_values[2]))
    ch4_data.append(float(lakeshore_values[3]))
    ch5_data.append(float(lakeshore_values[4]))
    ch6_data.append(float(lakeshore_values[5]))
    ch7_data.append(float(lakeshore_values[6]))
    ch8_data.append(float(lakeshore_values[7]))
    n2_data.append(float(n2_response))
    
    ax1.clear()
    ax2.clear()
    
    ax1.plot(t_data, ch1_data, label=lakeshore_channel_locations[0])
    ax1.plot(t_data, ch2_data, label=lakeshore_channel_locations[1])
    ax1.plot(t_data, ch3_data, label=lakeshore_channel_locations[2])
    ax1.plot(t_data, ch4_data, label=lakeshore_channel_locations[3])
    ax2.plot(t_data, n2_data, label=r'liquid level', linestyle=':', c='k')
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax2.legend()
    plt.tight_layout()
    
    """Pause between readings"""
    #time.sleep(acquisition_time)

if __name__ == "__main__":
    """As long as this scrip is running and/or the connection to the serial device is open""" 
    model_1700.write('CONFigure:N2:UNIT CM\n'.encode(encoding = 'UTF-8'))
    while (lakeshore.isOpen() & model_1700.isOpen()):
        read_all()    
    #ani = FuncAnimation(fig, read_all, interval=10)
    #plt.show()
    #plt.savefig('sensors_' + timestamp + '.png')
    output_file.close()
