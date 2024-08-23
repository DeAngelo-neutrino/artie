# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import os
import math
import sys
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


    
os.chdir("/Users/dwooley/Desktop/Python_programs/")


#we need to create varibles that are assoicated with each channel so we can plot, (This has been complete.)
#This creates an array/ column of data
file_name = "log_all_chan_5_9-15_5_57.txt"
time_argon = np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(0))   
target_top = np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(1))   
target_side = np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(2))   
target_bottom = np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(3))   
front_plate= np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(4))  
N2_level= np.genfromtxt(file_name,skip_header=9, delimiter=",", usecols=(9))   
 

#print(time_argon)
#this plots the speific data we want to plot

plt.figure(1)
plt.plot(time_argon,N2_level,marker=".",markersize=3)
plt.title("Time vs Argon_levels")
plt.xlabel("Time [s]")
plt.ylabel("Percentage filled [cm]")
plt.show()


"""
plt.figure(2)
plt.plot(time_argon,target_top,color='red')
plt.title("Time vs Target_top")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.show()

"""





#now we want to try to do a average we are gonna try using pandas

df = pd.read_csv(file_name, skiprows=9, sep=',' ) # we have to use , as delimitter and we use skiprows to ignore the first rows before the data


#df = df.iloc[9:] #this removes all of the stuff in the file before getting to the good data

#link to rolling window
"https://campus.datacamp.com/courses/manipulating-time-series-data-in-python/window-functions-rolling-expanding-metrics?ex=1#:~:text=Window%20functions%20in%20pandas,-00%3A00%20%2D%2000&text=In%20particular%2C%20window%20functions%20calculate,of%20the%20original%20time%20series."

#print(df.columns) #this shows column names



window_size = 900 #900 was intial value

time_column = df['Time (s)']
N2_level_dataseries = df['N2 Level']
N2_average = N2_level_dataseries.rolling(window = window_size).mean() #creates an average of 30 ( look up what is a rolling average do?)
#print(time_column)
#print("This is the average with a window of 30" , N2_average)


plt.figure(3)
plt.plot(N2_level_dataseries, 'k-', label='Original')
plt.plot(N2_average, 'g-', label='Original') 
plt.ylabel("Argon Level [cm]")
plt.xlabel("Time[s]")
plt.show()


#link to volume of a hallow sphere 
"https://testbook.com/maths/volume-of-hollow-cylinder#:~:text=Volume%20of%20Hollow%20cylinder%20%3D%20Volume,”%20and%20height%20“h”.&text=So%2C%20the%20derived%20formula%20for,3D%20shapes"

#link to video for integration of spefici problem
"https://www.youtube.com/watch?v=PSlsj0IP8R8"


#math calculation

lower = 2.4892 #cm #.98 inches
upper = 12.22248 #cm  #4.812 #inches
h_ = upper
R_2 = upper
R_2_cm = 12.22248 #cm
L= 171.958 #normally this is 67.7 inches  but we convert to cm so it is 
v_h_0 = 15754.5233 #cm^3     #961.4 inches ^3
R_1 = 10.16#cm #normally 4 inches
R_1_cm = 10.16 #cm
h__lower = R_1
#what is v_R_1
twice_length = 2*L
h_zero = 2.4892#cm #.98 #inches







upper_limitcalculation =   ((1/2)* (np.sqrt(R_2*R_2 - h_*h_))*h_) + ((1/2)*R_2*R_2*np.arcsin(h_/R_2))
lower_limit_calculation =   ((1/2)* (np.sqrt(R_2*R_2 - h__lower*h__lower))*h__lower) + (1/2)*R_2*R_2*np.arcsin(h__lower/R_2)
full_integration = upper_limitcalculation-lower_limit_calculation

only_R2 =  2*L *full_integration # this is correct we cross check with wolfram and excel
#print("only R_2 in integral calculation",only_R2,"inches^3")


# we redefine varibles
h_ = R_1
h__lower =lower
upper_limitcalculation_1 =   ((1/2)* (np.sqrt(R_2*R_2 - h_*h_))*h_) + (1/2)*R_2*R_2*np.arcsin(h_/R_2)  #the reason we are getting an error is because of thr R_2^2 - h_ ^2 part this is giving us a negitive number
lower_limit_calculation_1=   ((1/2)* (np.sqrt(R_2*R_2 - h__lower*h__lower))*h__lower) + (1/2)*R_2*R_2*np.arcsin(h__lower/R_2)
part_integration_1 = upper_limitcalculation_1-lower_limit_calculation_1


Part_1_volume = 2*L *part_integration_1 
#print(Part_1_volume,"part_1 volume")



h_ = R_1
h__lower =lower
upper_limitcalculation_2 =   ((1/2)* (np.sqrt(R_1*R_1 - h_*h_))*h_) + (1/2)*R_1*R_1*np.arcsin(h_/R_1) #the reason we are getting an error is because of thr R_2^2 - h_ ^2 part this is giving us a negitive number
lower_limit_calculation_2=   ((1/2)* (np.sqrt(R_1*R_1 - h__lower*h__lower))*h__lower) + (1/2)*R_1*R_1*np.arcsin(h__lower/R_1)
part_integration_2 = upper_limitcalculation_2-lower_limit_calculation_2


Part_2_volume = 2*L *part_integration_2
#print(part_integration_2,"part 2 volme")







#we need to create functions that do the integral with different h values , because right now the h is static.
#h will be whatever we call our argon array 
# link to integral
#https://en.wikipedia.org/wiki/List_of_integrals_of_irrational_functions





def all_togeterRs(h_n2):
   # h = N2_level
    #now we need to create a equation that takes h as the input
    #the sqrt functinos need to be np and not math.
    return (((1/2)* (np.sqrt(R_2**2 - h_n2**2))*h_n2) + ((1/2)*R_2**2 * np.arcsin(h_n2/R_2))) - ((((1/2)* (np.sqrt(R_2**2 - h_zero**2))*h_zero) + ((1/2)*R_2**2 * np.arcsin(h_zero/R_2))) )  - ( (((1/2)* (np.sqrt(R_1**2 - h_n2**2))*h_n2) + ((1/2)*R_1**2 * np.arcsin(h_n2/R_1))) - (((1/2)* (np.sqrt(R_1**2 - h_zero**2))*h_zero) + ((1/2)*R_1**2 * np.arcsin(h_zero/R_1)))  ) 
#above calculation is in inches



df['test'] = twice_length * all_togeterRs(N2_average)
#print("hi",new_way.head(50))


df['volume(Argon)'] =  v_h_0+(df['test']) + twice_length*full_integration #this is in inches^3
volume_function_as_height = df['volume(Argon)']



lower_parsed_data = 6500 # this is the lower limit of where the data starts #previous value was 3000, for spline this only works for 7000 to 10000 with window size 900
upper_parsed_data = 9500 #this is the upper limit of where the data ends


data_we_use = volume_function_as_height.iloc[lower_parsed_data:upper_parsed_data]



print("max volume",data_we_use.head())



plt.figure(4)
plt.plot(volume_function_as_height, 'r-', label='Original')
#plt.plot(N2_average, 'g-', label='Original') 
plt.ylabel("Argon Level volume [in^3] as a function of height")
plt.xlabel("Time[s]")
plt.title(" Time vs  Argon Level volume [in^3] as a function of height")
plt.show()






"""
Now we need to do a partial derivative of the voulme as a function of time with respect to time 
or in math form
∂v(t)/∂t 
∂v(t)/∂t  = (∂v/∂t)(t)
"""




"""
diff_of_v_argon = np.diff(volume_function_as_height)
diff_of_time = np.diff(time_column)
dydx_0 = diff_of_v_argon/diff_of_time
print(len(dydx_0))
#df['boil off rate'] = dydx_0
#print("This is the of differential of volume of argon with respect to time",diff_of_v_argon/diff_of_time)
"""





plt.figure(5)
plt.plot(data_we_use, 'r-', label='Original')
#plt.plot(N2_average, 'g-', label='Original') 
plt.ylabel("volume as a function of height selected data [in^3]")
plt.xlabel("Time[s]")
plt.title(" Selected Data Chart 6500s to 10000s vs Volume[in^3]")
plt.show()




#N2_level_dataseries = df['N2 Level']
Time_average_dataseries = time_column.rolling(window = window_size).mean() #creates an average of 30 ( look up what is a rolling average do?)
df['Time Average'] =  Time_average_dataseries
time_average = df['Time Average']
parsed_time_average = time_average.iloc[lower_parsed_data:upper_parsed_data] #to ensure the data arrays are the same size
#print(print(df['Time'].head(75)))





#gradient explanation
#https://stackoverflow.com/questions/24633618/what-does-numpy-gradient-do


#another way
dy= np.gradient(data_we_use) #parsed argon level data 
dx = np.gradient(parsed_time_average) #parsed time data
dydx = dy/dx





"""
degree = 2
coefficients = np.polyfit(parsed_time_average, dydx, degree)
poly = np.poly1d(coefficients)
# Generate fitted values
fitted_values = poly(parsed_time_average)

plt.figure(9)
plt.plot(parsed_time_average, fitted_values, color='red', label=f'Polyfit (degree {degree})')
#plt.scatter(parsed_time_average, dydx, label='Data')

plt.xlabel('Parsed Time Average')
plt.ylabel('dydx')
plt.legend()
plt.show()
"""



"""
# Fit a spline to the data
spline = UnivariateSpline(parsed_time_average, dydx, s=1)

fitted_values = spline(parsed_time_average)

coefficients = spline.get_coeffs()
equation = "Spline equation:\n"
if coefficients.size >0:
    equation += "f(x) = " + " + ".join(f"{coeff:.2f}*x^{i}" for i, coeff in enumerate(coefficients))
else:
    equation += "No coefficients available."

# Plot the results
plt.figure(9)
plt.scatter(parsed_time_average, dydx, label='Data')
plt.plot(parsed_time_average, fitted_values, color='red', label='Spline Fit')
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
         fontsize=5, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

plt.xlabel('Parsed Time Average [seconds]')
plt.ylabel('dydx [inch^3]/[seconds]')
plt.legend()
plt.grid()
plt.show()
#sometimes the reason why the spline is messing up is because it can not plot the discontinous function
"""


plt.figure(6)
plt.plot(parsed_time_average,dydx, 'k-', label='Original')
plt.ylabel("dy argon level [inches^3]")
plt.xlabel("dx Time[s]")
plt.title(" boil of rate[gradient method]")
plt.show()







#calculate heat load 
density_liquid_argon = 1410 #kg/m^3
h_vap = 162.8 #kj/kg
centimeters_cube_to_meters_cubed = 1e-6
Q_heat = dydx * centimeters_cube_to_meters_cubed * h_vap*density_liquid_argon #need to use dydx and not dydx_0 because different sizes of array.

#print(len(Q_heat))
#print(Q_heat)


plt.figure(7)
plt.plot(parsed_time_average,Q_heat , 'k-', label='Original')
plt.ylabel("Heatload [KJ]/[s] ")
plt.xlabel("Time[s]")
plt.title("Heat load plot")
plt.show()


#mean_heat_load = Q_heat.mean() 
mean_heat_load = np.nanmean(Q_heat)  # Use numpy's nanmean function to ignore nan values in the array
print(mean_heat_load,"mean heat load in [kilojolues]/[second]")
print("if we convert to [joules]/[second] we get",mean_heat_load*1000,'\n')
#print(Q_heat)


#print(f"Min parsed_time_average: {np.min(parsed_time_average)}")
#print(f"Max parsed_time_average: {np.max(parsed_time_average)}")
#print("Spline Coefficients:", coefficients)





def exponential_decay(x, A, B, C):
    return  A* np.exp(-B * x) + C


x_data = parsed_time_average  
y_data = data_we_use 





# Initial guess for parameters A, B, and C


B = 1*10**-4
initial_guess = [max(y_data),B, min(y_data)]

# Perform the curve fitting
popt, pcov  = curve_fit(exponential_decay, x_data, y_data, p0=initial_guess)

# Extract the optimal parameters
A_opt, B_opt, C_opt = popt

# Generate y values using the fitted parameters
y_fit = exponential_decay(x_data, *popt)


# Plot data and fit
annotation_text = (f'A = {A_opt:.2f}\n' f'B = {B_opt:.2e}\n'     f'C = {C_opt:.2f}')
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'g-',label='Data')
plt.plot(x_data, y_fit, label='Exponential Decay Fit data', color='blue')
plt.xlabel('X')
plt.ylabel('Y')
#plt.plot(parsed_time_average,data_we_use, 'r-', label='Original')
plt.title('Exponential Decay Fit')
plt.legend()
plt.text(.25, 0.95, annotation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='black')
plt
plt.show()

# Print the optimal parameters
print(f"Optimal parameters:\nA = {A_opt}\nB = {B_opt}\nC = {C_opt}",'\n')



y_fit = np.gradient(y_fit)
dx = np.gradient(parsed_time_average) #parsed time data
dydx_new = y_fit/dx







plt.figure(figsize=(10, 6))
plt.plot(parsed_time_average,dydx_new, 'k-', label='Original')
plt.ylabel("dv/dt argon level [inches^3]/[Seconds]")
plt.xlabel("dt Time[s]")
plt.title(" boil of rate[gradient method] with decayfit")
plt.show()
plt.legend()
#plt.text(.25, 0.95, annotation_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='black')
plt.show()



#calculate heat load 
density_liquid_argon = 1410 #kg/m^3
h_vap = 162.8 #kj/kg
centimeters_cube_to_meters_cubed = 1e-6
Q_heat = dydx_new * centimeters_cube_to_meters_cubed * h_vap*density_liquid_argon #need to use dydx and not dydx_0 because different sizes of array.


print("Heat_load",Q_heat[0])
print("time",parsed_time_average[lower_parsed_data],'\n')




plt.figure(figsize=(10, 6))
plt.plot(parsed_time_average,Q_heat , 'k-', label='Original')
plt.ylabel("Heatload [KJ]/[s] ")
plt.xlabel("Time[s]")
plt.title("Heat load plot by using decay fit data")
plt.show()



#mean_heat_load = Q_heat.mean() 
mean_heat_load = np.nanmean(Q_heat)  # Use numpy's nanmean function to ignore nan values in the array
print(mean_heat_load,"mean heat load in [kilojolues]/[second]")
print("if we convert to [joules]/[second] we get",mean_heat_load*1000,'\n')
#print(Q_heat)


mean_boil_off_rate = np.nanmean(dydx_new)  #david wants to see
print("this is the mean of dv/dt",mean_boil_off_rate)





