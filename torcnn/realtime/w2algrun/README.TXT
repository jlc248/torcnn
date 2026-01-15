# REST 2002
#  -- Robert ES Toomey  Robert.Toomey@noaa.gov
#
# Generic process forever runner and logger
#
# The idea is to monitor and run generic processes
# on a linux system similiarly to inetd, but for multiple
# individual users
#
#

PControl -- Controller of a list of PGroups (Process Groups)
PGroup -- Controls a list of PMan (Process Managers)
PMan -- Controls a list of N unix processes

w2logger.pl handles rotating logs for a process
w2run.pl handles watching and restarting dead processes


Quick Start:
1. Run autogen.sh to compile perl code, or copy w2logger.pl to w2logger
   and w2run.pl to w2run to just use as scripts
2. Edit w2alg.conf to have what you want.
3. Use the w2alg command to get detailed help.  Basically, you use
   commands in the form:

   w2alg start ktlx netssap

   Where ktlx is a 'group', netssap is a named 'manager'.  See the 
   w2algExample.conf for details.


##############################################################################
# Log format:

Log type 1:
(shared log for runners) w2runlog format:

  (1) Start(identifier) PID: Number
     -- w2run started a child process with id 
  (2) Stop(identifier) PID: Number
     -- w2run force stopped a child process with id 
  (3) DIED(identifier)
     -- Child with identifier returned finished or died
     -- w2run force stopped a child process with id 

Log type 2:
(single log per identifer)
   ****Machine information on start
   ****EXELINE: command line running
   --All stdout, stderr output from process
