#!/usr/bin/perl -w

#
# Display w2alg usage (separate file to reduce module sizes)
#
# usage.pl exename {list}
use strict;

Main: {

  # Read args, which is just exe name and what to help with
  my ($e, $what);
  if ($#ARGV <0){ $e = "w2alg"; }else{ $e = $ARGV[0]; }
  if ($ARGV[1]){ $what = $ARGV[1]; }else { $what = ""; }

if ($what eq ""){
print<<ENDOFUSAGE;
Usage: $e

Examples:
  $e start all           # Start all managers for all groups in conf
  $e start KLTX          # Start all managers for group KLTX
  $e start KLTX SSAP     # Start SSAP for group KLTX 
                          (Need SSAP, or PROCESS-SSAP as manager)

  $e stop all            # Stop all managers for all groups in conf
  $e stop KLTX           # Stop all managers for group KLTX
  $e stop KLTX SSAP      # Stop SSAP for group KLTX (if no dependences)

  $e restart KLTX        # Stop then Start group KLTX

  $e loginfo KTLX        # Show log path for each manager in KTLX
  $e loginfo KTLX SSAP   # Show log path for manager 
  $e loginfo mainlog     # Show path to shared log

  $e status all ssap     # Show run status for managers named SSAP
                         # within all groups

  $e rotate KLTX         # Rotate logs for KLTX (can be used while
                             running)
  $e rotate KLTX SSAP    # Rotate logs for group KLTX, manager SSAP
  $e rotate mainlog      # Rotate shared w2run log for all runners

  $e rotate KLTX         # Rotate logs for KLTX (can be used while
                             running)

  $e clear KLTX          # All rotate options, but truncate logs
  $e clear mainlog       # Clear out shared log

  $e list                # Lists various information (see below)

Note that processes are always attempted to be started in the order they
appear within the config file.  If a group and manager are not in the 
configuration file then the command returns an error.

COMMANDS:

RUNNING COMMANDS:

  start | run { all | pGroup [pManager] }       
        - Start up  all or a given pGroup

  stop | kill { all | pGroup [pManager] }   
        - Stop 

  reload | restart { all | pGroup [pManager] } 
        - Reload.  Calls stop for all managers, then start for all
                Global is always stopped last and started first
  status { all | pGroup [pManager] }
        - Show run status for given group/manager

LOG RELATED COMMANDS:

  rotatelog | rotate | rotatelogs 
     { all | pGroup [pManager] | mainlog }
        - Force log rotation

  clearlog | clear | clearlogs { all | pGroup [pManager] | mainlog }
        - Clear log files

  loginfo { all | pGroup [pManager] | mainlog }
        -- Show log path

  tail { pGroup pManager | mainlog }
        -- Tail log


ENDOFUSAGE
}

if (($what eq "") || ($what eq "list")){
print<<ENDOFLISTHELP;
INFORMATION COMMANDS:

  list { all | paths | run | conf | logs }

     $e list all or "" --> PATHS, CONF, RUN
     $e list paths     --> Show path information
     $e list run       --> Show current runners/loggers
     $e list conf      --> Parse current configuration file 
     $e list logs      --> Show current log locations

ENDOFLISTHELP
}

}
