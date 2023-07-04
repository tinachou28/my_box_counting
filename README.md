# Box_Counting
Some codes to count particles in boxes and calculate statisitcs 

To run the code with a linux OS, first navigate to the working directory in a treminal and compile the C++ module with `make`. If you're running the code on MAC_OS, then rename the file `Makefile` to `Makefile_Linux` and the file `Makefile_MACOS` to `Makefile`. 

To perform the data analysis, modify the file `Fast_Box_Stats.py` for your specific example and run `python Fast_Box_Stats.py` to execute the code. 

The MATLAB file `timescale_integral.m` processes the data computed using `Fast_Box_Stats.py` by computing the timescale integral. 

# Pure python box counting
To run the pure python code, simply modify the main `Fast_Box_Stats_NoCpp.py` and run with `python Fast_Box_Stats_NoCpp.py`
