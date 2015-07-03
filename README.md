# Flow tracing of electricity in New Zealand

This repository contains code to run the power flow tracing routine that is used as part of the Electricity Authority's proposed Transmission Pricing Methodology (TPM) as published by the Authority in an options paper on 16 June 2015.

Transmission asset usage can be determined using power flow tracing and is being used here as an example of the Market Design of a Transmission Pricing Methodology.  It is used to distinguish between assets that are used by many (called interconnection assets) and assets that are used or shared among only a few (currently called "deep" connection assets).

**NOTE: this is a live repository and will be updated over the coming weeks with additional code as I make it available.  It is similar, but not the same as, the code based used for the TPM work.  Further improvements are planned in the future (DJH, 3 July, 2015)
Have made changes to the post processing, process - see below- still waiting for OK on RAB file publish.**

## Inputs:

The only required inputs for this calculation are the MW power flows and directions on each transmission element, the load at load buses and the generation at generation buses. Each transmission element must have a unique identifier (i.e., a TO and FROM bus) so the tracing methodology is aware of the network topology.
 
These inputs are the outputs of a solved SPD or vSPD market model.  So the tracing methodology follows the market solve and is applied at trading period level.

In this case the input data is from a Concept Consulting vSPD run and in a slightly altered format to the output data found from a generic vSPD run.  

The input data for 3 years of data runs to many GB in size.  It has been split into manageable monthly files which can be downloaded from the EA EMI website at:

http://bit.ly/1eTEd9o

A sample of the input data can be found within this repository at /data/input along with a fuller description of the input data which also includes mapping files required for the tracing routine. 


## Outputs

Through the use of what Bialek[1] described as the proportional sharing principle, Flow Tracing is able to calculate the share of usage of transmission assets.  It is able to allocate usage in two directions: 

  - an "upstream" trace; where transmission asset usage is allocated to grid connected generators, and,
  - a "downstream" trace; where transmission asset usage is allocated downstream to demand/load. 
  
As discussed, this is achieved at both trading period level and at GXP/GIP level.  It is a simple matter to then group and sum this GXP/GIP level data to a participant level.

Once ./trace.py is run, the output files should (as of 29/6/2015) appear in the data/output directory.  If this directory does not exist it should appear with the following sub-directories.
```
data/output/tp  # trading period mean
data/output/d  # daily mean
data/output/m  # monthly mean
data/output/y  # annual mean
data/output/t  # total mean
```

The output files are saved as the mean over all trading periods during each month.  Monthly output files are saved of the following form:
```
- td_YYYYMM.csv "downstream" results that allocate circuits and transformers to off-take/demand GXPs.
- tu_YYYYMM.csv "upstream" results that allocate circuits and transformers to generation GIPs. 
- sd_YYYYMM.csv "downstream" results allocating substations to off-take/demand GXPs. 
- su_YYYYMM.csv "upstream" results allocating substations to generation GIPs. 
```

## Post-processing trace output data

The following python and iPython notebook files are provided for post-processing.
```
trace_post_processing_1.py 
trace_post_processing_hvdc.ipynb  
trace_post_processing_2.ipynb
```
trace_post_processing_1.py returns the mean of the daily data over months, years and the total three year period and adds the Herfindahl-Hirschman Index (HHI).  It also strips additional data from the raw trace output for the substation trace results (some of this should probably should have been part of trace.py...)

trace_post_processing_hvdc.ipynb is an attempt to look at the HVDC.  The HVDC is unique and is modelled in vSPD as four circuits, two northward flowing and two southward flowing.   Because of this, ideally these circuits need to be combined into one and the HHI calculated for the HVDC as per any other AC circuit.  We have not done this so these results are representative and the HVDC is *not* intended to be traced as a deep connection asset. 

trace_post_processing_2.ipynb (initial version published) adds Transpower's Regulated Asset Base (RAB) for interconnection assets to each of the four total output csvs - waiting for confirmation on publishing the RAB data.  This is currently in iPyhton notebook format.  


## Installation instructions

As of 9/6/2015, this code has only been tested on a Linux machine running Ubuntu 15.04.  We may test on a windows machine in the future.
To run this code requires an installation of python 2.7 with the python modules as indicated in the requirements.txt file. 

The best way to run this code is from within a virtual environment using the virtualenv python package.  This allows the required dependencies in the requirements.txt file to be installed independently from any other python installation on the host machine.  

You can install virtualenv via pip:
```
$ pip install virtualenv
```
To clone this repository, open a terminal window and type/paste the following commands;
``` 
$ git clone https://github.com/ElectricityAuthority/tracing.git
$ cd tracing
```
Once the repository has downloaded, we can set up a virtual environment in the local directory called env by typing: 
```
$ virtualenv env
```
We activate the virtual environment, by typing;
```
$ source env/bin/activate
```
Running the trace now would cause errors as the trace.py is dependent on a number of other python modules.  
To solve this we install the following packages into the virtual environment by typing:
```
$ pip install numpy
$ pip install pandas
$ pip install psutil 

```
We should now be able to run the trace over the sampled data supplied.

To run the trace, issue the following command:
```
$ ./trace.py
```
All going well, the trace should start and you should see confirmation in the terminal. It should take between 1 and 2 seconds to complete a trace on each trading period.  
Daily output files will be generated for one month. 

Adding Input files from the EMI website into the /data/input/vSPDout directory and editing trace.py (test_limit_min and test_limit_max) should allow the program to run over other time periods. 



## Historic tracing of vSPD daily output file

A slightly altered version of this code has been written to take advantage of the daily vSPD output files and this can be used for tracing historic output.  A slight modification to the output is required and for this reason a small Gams script is also required to output additional bus/node mapping data.  The EA may also publish this code in the future.


## Additional background information

Electricity tracing was formally described by Janusz W. Bialek in his classic 1996 paper titled Tracing the flow of electricity [1].

The method has since been applied in NZ by the Electricity Commission in 2010 as part of the on-going Transmission Pricing Methodology.  To aid in understanding the methodology, a paper was written at the time and is available in the docs directory.  It is also still available via the EA website at:

https://www.ea.govt.nz/dmsdocument/7123

More recently this methodology has been updated (this work), using open source software, and as part of the more recent Transmission Pricing options work currently being conducted by the Market Design team within the Electricity Authority.

For further information, comments, edits, etc, contact:

Contact: emi@ea.govt.nz

[1] J. Bialek, Tracing the flow of electricity, IEE Proc.-Gener. Transm. Distrib., Volume 143, No. 4, Page(s):313–320, July 1996.

