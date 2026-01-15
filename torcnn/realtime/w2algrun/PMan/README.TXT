REST 
  -- Robert ES Toomey  Robert.Toomey@noaa.gov

These are the modules which will handle a particular algorithm or set of processes.
The name of the module must be ALG.pm, where ALG is in full caps.

You can then call "start groupname modulename" to run the alg for a particular radar

Note, if you just want to run a single process, you can use the PROCESS.pm module
(if you want to run a single command line)

Fairly complete modules:
  PROCESS.pm  -- Single generic one a single process module
  LOG.pm      -- Single window to watch a log file
