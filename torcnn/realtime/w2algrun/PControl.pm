#--------------------------------------------------------------------------
#
# REST 2002
#
# PControl --Class which owns multiple PGroup objects
#
# PControl --n--> PGroup --n--> PMan --n--> runForevers
#
#--------------------------------------------------------------------------
package PControl;

# Find where the w2alg bin is, independent on where it's executed from,
# this is our home path. For example, if w2alg is in /usr/bin then 
# $RealBin = /usr/bin
use FindBin qw($Bin $Script $RealBin $RealScript);
use lib "$RealBin";
use Cwd;

use PGroup;
use PMan;
use PLogUtil;
use PDateStamp;

# All of these are currently in the w2alg directory
$PControl::CONFIG  = "w2alg.conf";           # The global config file
$PControl::LOG     = "w2run#.log";           # The global log file
$PControl::RUN     = "$RealBin/w2run";       # EXE name for runforever
$PControl::LOGGER  = "$RealBin/w2logger";    # EXE name for logger
$PControl::PROCER  = "$RealBin/w2procwatch"; # EXE name for logger
$PControl::REALBIN = "$RealBin";             # Bin for all exes
$PControl::TAIL    = "tail -f";              # tail to use
$PControl::HISTORY = "$ENV{HOME}/.w2alghist";# history file to use
$PControl::LASTRUN = "$RealBin/wid.txt";     # Last process id started


sub new
{
   my ($type) = @_;
   my ($self) = { };
   $self->{gPGroup} = {};       # GLOBAL manager, if exists
   $self->{PGroups} = "";       # Loaded PGroups
   bless $self, $type;
}

#--------------------------------------------------------------------------
# initialize
#
# -- Read command line arguments and do what needs to be done
#
sub initialize {
  my $self = shift;

  # Make sure // are gone. Some os put an end /, some don't
  # Note all submodules will use us so this needs to be done first
  # before anything else
  $PControl::RUN =~ s/\/\//\//g;
  $PControl::LOGGER =~ s/\/\//\//g;
  $PControl::PROCER =~ s/\/\//\//g;
  $PControl::HISTORY =~ s/\/\//\//g;
  $PControl::LASTRUN =~ s/\/\//\//g;

  if ($#ARGV >= 0){
    my $param1 = "\U$ARGV[0]";
    my $param2 = "\U$ARGV[1]";
    my $param3 = "\U$ARGV[2]";

    my $handled = 1;
    if ($param1 eq "LOGINFO"){
      $self->HandleLogCommand("LOGINFO", $param2, $param3);
    }elsif ($param1 eq "LIST"){
      $self->listInfo($param2);
    }elsif ($param1 eq "STATUS"){
      print"Identifier (group-name-process) => Processes (See STATUS.TXT)\n";
      print"---------------------------------------------------------------\n";
      $self->updatePGroups("STATUS", $param2, $param3);
      print"***COMPLETED STATUS\n";
    }elsif (($param1 eq "START") || ($param1 eq "RUN")){
      $self->updatePGroups("START", $param2, $param3);
      print"***COMPLETED INITIALIZATION\n" if ($#ARGV == 1);
    }elsif (($param1 eq "STOP") || ($param1 eq "KILL")){
      $self->updatePGroups("STOP", $param2, $param3);
      print"***COMPLETED SHUTDOWN\n";
    }elsif (($param1 eq "RESTART") || ($param1 eq "RELOAD")){
      $self->updatePGroups("RESTART", $param2, $param3);
      print"***COMPLETED REBOOT\n";
    }elsif ($param1 eq "HEARTBEAT"){
      $self->updatePGroups("HEARTBEAT", $param2, $param3);
    }elsif ($param1 eq "TAIL"){
      $self->HandleLogCommand("TAIL", $param2, $param3);
    }elsif (($param1 eq "ROTATE") || 
            ($param1 eq "ROTATELOGS") ||
            ($param1 eq "ROTATELOG")){
      $self->HandleLogCommand("ROTATE",$param2, $param3);
      print"***COMPLETED LOG ROTATION\n";
    }elsif (($param1 eq "CLEAR") || 
            ($param1 eq "CLEARLOGS") ||
            ($param1 eq "CLEARLOG")){
      $self->HandleLogCommand("CLEAR",$param2, $param3);
      print"***COMPLETED LOG CLEARING\n";
    }else{ $handled = 0; }

    # Write simple history file, simple rotation
    if ($handled) { 
      if (-s $PControl::HISTORY > 10000){  # two 10k logs
        `mv $PControl::HISTORY ${PControl::HISTORY}1`;
      }
      if (open(HISTORY, ">>$PControl::HISTORY")){
         print HISTORY PDateStamp::dateStamp();
         print HISTORY "$param1 $param2 $param3\n";
      }
      close(HISTORY);
      return;
    }
    # --- end of simple history logger
  }

  # Get name we are running as and pass to help
  my ($e) = ($0 =~ /([^\/]*)$/);
  exec "$RealBin/usage.pl $e";
}

#--------------------------------------------------------------------------
# HandleLogCommand
# -- Handles each groups logs, than the global shared log
# $what eq "ROTATE" --> Rotate logs
# $what eq "CLEAR" --> Clear logs
# $what eq "TAIL" --> Tail log (spawn tail)
#  Takes special 'MAINLOG' group to refer to main run log
sub HandleLogCommand {
  my $self = shift;
  my ($what, $group, $manager) = @_;

  # Turn blanks into all requests "w2alg start ktlx" ==> 
  # "w2alg start ktlx all"
  $group = "ALL" if ($group eq "");
  $manager = "ALL" if ($manager eq "");

  # If not MAINLOG, pass to update as normal command, otherwise
  # force load of GLOBAL env (We need W2 log parameters from conf)
  if ($group eq "MAINLOG"){ $self->loadGlobalEnv();
  }else{
    # In tail case, we can't tail multiple files..or can we?
    if ($what eq "TAIL"){

     #if (($group eq "ALL") || ($manager eq "ALL")){
     #  print "Error: Log to tail is ambiguous.  Need group and manager\n";
     #  print "EX: w2alg tail ktlx ldm2netcdf\n";
     #  exit(1);
     #}
    }
    $self->updatePGroups($what, $group, $manager);
  }

  # Now handle the special MAINLOG case:
  # Mainlog doesn't belong to any group.  So we have to handle its commands
  if (($group eq "MAINLOG") || ($group eq "ALL")){
    my ($logFile, $logNum, $logSize);

    PLogUtil::getLogInfo($PControl::LOG, "w2", \$logFile, 
       \$logNum, \$logSize);

    if ($what eq "LOGINFO"){
      $logFile =~ s/#/1/;
      print "Current main log:   $logFile\n";
    }elsif ($what eq "ROTATE"){
      print "Rotating shared ---> $logFile\n";
      PLogUtil::rotateLogs($logFile, $logNum);
    }elsif ($what eq "CLEAR"){
      print "Clearing shared ---> $logFile\n";
      PLogUtil::clearLogs($logFile, $logNum);
    }elsif ($what eq "TAIL"){
      $logFile =~ s/#/1/;
      print "Tailing shared ---> $logFile\n";
      my $tail = $PControl::TAIL;
      exec "$tail $logFile";
    } # else what??
  }
}

#--------------------------------------------------------------------------
# statusAllPGroups
# -- Dumps running status of all runners and loggers found
# "w2alg list run" or "w2alg status"
#
sub statusAllPGroups {
  my $self = shift;
  my (@pGroupsToCheck);
  my ($line, $exe, $leftSize, $rightSide, $logger);
  my ($r,$a,$p,$arg, $col);

  $exe = $PControl::RUN;
  $logger = $PControl::LOGGER;

  # Match up found processes in system that are running
  $col = $ENV{COLUMNS}; $ENV{COLUMNS} = 500;
  if (open (PS_PIPE, "ps -A -o pid -o args |")){
    print "Active forever runners detected:\n";
    while ($line = <PS_PIPE>){

       # Broke down the match for my sanity
       if ($line =~ /(.*)$exe(.*)/){   # Pull out processes first
         $leftSide = $1; $rightSide = $2;
         if ($leftSide =~ /([0-9]+)/){ $pid = $1; }
         else {$pid = "UNKNOWN";}
         if ($rightSide =~ /([^ -]+)-([^ -]+)-([^ -]+)(.*)/){
    print "  PID $pid  Group $1, Name $2, (Process $3)\n";
         }
      }
    }

    close (PS_PIPE);
  }

  # Find loggers now
  if (open (PS_PIPE, "ps -A -o pid -o args |")){
    print "Active loggers detected:\n";
    while ($line = <PS_PIPE>){
       if ($line =~ /(.*)$home$logger(.*)/){   # Pull out processes first
         $leftSide = $1; $rightSide = $2;
         if ($leftSide =~ /([0-9]+)/){ $pid = $1; }
         else {$pid = "UNKNOWN";}
         print "  PID $pid $rightSide\n";
      }
    }
    close (PS_PIPE);
  }
  $ENV{COLUMNS} = $col;
}

# Commands runnable by the GUI.  Humans can run them,
# but they won't be pretty...
# disabled for now... probably going to dynamically
# load the XML perl module.
sub guiInterface {
  my $self = shift;
  my ($param1, $param2) = @_;
  my $buffer="";

  if ($param1 eq "GROUPS"){
    my (@pGroupsInConfig) = $self->readConfig();
    foreach (@pGroupsInConfig){
      $buffer.="  <group>$_</group>\n";
    }
    print<<ENDOFXML1;
<?xml version="1.0"?>
<groupinfo>
$buffer</groupinfo>
ENDOFXML1
  }
}

#--------------------------------------------------------------------------
# listInfo
#  -- List information about setup.  
#
# Broken up into sections in case it gets really long
# which is easy with dozens of configurations

sub listInfo {
  my $self = shift;
  my ($what) = @_;
  my ($output);
  my $all = (($what eq "ALL") || ($what eq ""));
  my $handled = 0;   # Did something get listed?

  # List path information ------------------------------------------------
  # w2alg list conf 
  if ($all || ($what eq "PATHS")){
    $handled = 1;
    my $config = $self->configFile();
  $output=<<ENDOFLIST;
Configuration file: $config
-Control script:    $RealBin/$RealScript
-Runner  script:    $PControl::RUN
-Logger  script:    $PControl::LOGGER
ENDOFLIST
  print $output;
  }

  # List configuration file information -----------------------------------
  # w2alg list run 
  if ($all){
    print 
"--------------------------------------------------------------------\n";
  }
  if ($all || ($what eq "CONF")){
    $handled = 1;
    print "Current config file reading:\n";
  # This needs work...also, reg expressions should match 
  # the 'real' thing..which it doesn't
  if (open(CONFIG, $self->configFile())){
    my $groupList;
    while($line = <CONFIG>){ 
      if ($line =~ /^[ \t]*group[ \t]*([^\n#]*)/i){
        $groupList = $1;
        $groupList =~ s/[ \t]//g;    # get rid of tabs and spaces
        if ($groupList =~ /,/){
          print "GroupList: '$groupList'\n";
        }else{
          print "Group: '$groupList'\n";
        }
        next; 
      }
      # technicaly, PControl shouldn't 'know' about manager
      if ($line =~ /^[ \t]*manager[ \t]*([^ \n\t#]*)/i) {
        $name = $1;
        $module = $name;
        if ($name =~ /(.*)-(.*)/){ $module = $1; $name = $2; }
        print "   Manager: PMan/$module.pm  ";
        print "Name: $name\n";
        next; 
      }
    }
  }
  close CONFIG;
  }

  # List log information ---------------------------------------
  # w2alg list logs
  if ($all){
    print
"--------------------------------------------------------------------\n";
  }
  if ($all || ($what eq "LOGS")){
    my $logDir;
    my $log = $PControl::LOG;

    # Send command to other loaded groups to display
    $self->loadPGroups("ALL", "ALL");
    $self->commandPGroups("LOGINFO");

    # Display MAINLOG info
    PLogUtil::getLogDir("w2",\$logDir);
    $log =~ s/#/1/;
    print "Current main log:   $logDir/$log\n";
    $handled = 1;
  }

  # List runner/logger information ---------------------------------------
  # w2alg list run 
  if ($all){
   print
"--------------------------------------------------------------------\n";
  }
  if ($all || ($what eq "RUN")){
    $handled = 1;
    $self->statusAllPGroups;
  }

  # Nothing handled, print usage
  if ($handled == 0){
    print "$what is not an available LIST option.\n";
    my ($e) = ($0 =~ /([^\/]*)$/);
    exec "$RealBin/usage.pl $e list";
  }
}

#--------------------------------------------------------------------------
# loadGlobalEnv(pGroup, pMan)
#     -- Load 'set' for global group.  Contains defaults needed
#
# loadPGroups(pGroup, pMan)
#     -- Load 
#     requestPGroups(pGroup, pMan, $loadGlobalMans)
#     -- does the work for loadGlobalEnv and loadPGroups
# updatePGroups(what, pGroup, pMan)
# commandPGroups(what)
#     -- send each PGroup we control with a command
#     what = command such as start, stop, restart/reload, rotate
#
# pGroup = all or the pGroup to start all algorithms for
# pMan = optional single manager to start
#
sub loadGlobalEnv {
  my $self = shift;

  # Load global env only in case where we require variables,
  # but not any objects
  $self->requestPGroups("GLOBAL","", 0);
  $self->setEnv();
}

sub loadPGroups {
  my $self = shift;
  my ($wantPGroup, $wantPMan) = @_;
  my ($loadGlobalMans);

  # Turn blanks into all requests "w2alg start ktlx" ==> "w2alg start ktlx all"
  $wantPGroup = "ALL" if ($wantPGroup eq "");
  $wantPMan = "ALL" if ($wantPMan eq "");

  # load global managers or just env?  We only want managers within
  # global if its an "ALL" request, or a direct "GLOBAL" request
  $loadGlobalMans = (($wantPGroup eq "ALL") | ($wantPGroup eq "GLOBAL"));

  $self->requestPGroups($wantPGroup, $wantPMan, $loadGlobalMans);
}

sub requestPGroups {
  my $self = shift;
  my ($wantPGroup, $wantPMan, $loadGlobalMans) = @_;

  my ($curGroupName, $aPGroup, @curPGroupList);
  my (%createGroups) = {};
  my ($firstGroup);
  my ($foundGroup) = 0;
  my ($foundManager) = 0;
 
  # Set up
  if ($wantPGroup eq "ALL") { $foundGroup = 1; };
  if ($wantPMan eq "ALL") { $foundManager = 1; };
  $aPGroup = undef; $curGroupName = undef; $firstGroup = 1; $found = 0;

  print "Using config file '";
  print $self->configFile();
  print "'\n";
  if (open(CONFIG, ($self->configFile()))){
    my $notdone = 1;
    while($notdone){

      # Read in a line
      if ($line = <CONFIG>){ }else {$notdone = 0; next;} chop $line;

      # Found comments or blanks lines
      if ($line =~ /^[ \t]*#/) { next; }  # skip full line comments
      if ($line =~ /^[ \t]*$/) { next; }  # skip empty lines

      # Found group
      if ($line =~ /^[ \t]*group[ \t]*([^\n#]*)/i){
        $curGroupName = $1;
        $curGroupName =~ s/[ \t]//g;    # get rid of tabs and spaces
        @curGroups = split /,/, $curGroupName;

        # For each group found
        @curPGroupList = ();
        foreach $curGroupName (@curGroups){

          # First group in file MUST be global
          if (($firstGroup) && ($curGroupName ne "GLOBAL")){
           print<<GLOBALWARN;
ERROR: Configuration file must contain GLOBAL as first group
First group is $curGroupName
GLOBALWARN
             exit(1);
          }
          $firstGroup = 0;

          # We found requested group name in file
          if ($curGroupName eq $wantPGroup){ $foundGroup = 1; }

          # -----------------------------------------------------
          # Possibly make new group, or pull from hash
          if (($wantPGroup eq "ALL") || ($curGroupName eq "GLOBAL") ||
              ($curGroupName eq $wantPGroup))
          {
             # Make group if not in hash (only one object per group)
             if (defined($createGroups{"$curGroupName"})){
                $aPGroup = $createGroups{"$curGroupName"};
             }else{
                $aPGroup = new PGroup ($curGroupName, $self); 
                # Keep group for later
                $createGroups{"$curGroupName"} = $aPGroup;
                if ($curGroupName eq "GLOBAL"){ 
                  $self->{gPGroup} = $aPGroup;
                }else{ push @{$self->{PGroups}}, $aPGroup; }
             }
             $aPGroup->initGroupConfig(); 
             push @curPGroupList, $aPGroup;

          }#else ignore this group
          # -----------------------------------------------------

        } # groups
      }else{

          # Ok..if we have a current grouplist, send line to groups
          foreach $aPGroup (@curPGroupList){
            my $flag;   # load managers, or just env settings?
            if ($aPGroup == $self->{gPGroup}){ $flag = $loadGlobalMans;
            #print "Loop ($flag) with $aPGroup (global) $line\n";
            }else{ $flag = 1; }
            if($aPGroup->handleConfigLine($line, $wantPMan, $flag) == 1){
            #print "Loop ($flag) with $aPGroup $line\n";
               $foundManager = 1;   # group responded
            };
          }
       # else we ignore this line
      }
    }
    close(CONFIG);
  }else{
    print "Could open config file\n";
  }

  # --------------------------------------------------------------------------- 
  # Error checking.
  #

  # Make sure we created global object, found group/manager.
  if (!($self->{gPGroup})){
    print "ERROR: MUST have group GLOBAL in your configuration file $_\n";
    exit(1);
  }else{
    my($machine) = `hostname`;
    chomp $machine;
    my($infile) = ${$self->{gPGroup}->{env}}{"MACHINE"};
    if ($infile eq $machine){
    }else{
      print "MACHINE is '$infile' and does not match hostname '$machine'.\n";
      print "set MACHINE \"$machine\" needs to be in the conf file\n";
      exit(1);
    }
  }
  if (!$foundGroup){
    print "ERROR: group '$wantPGroup' does not exist in your config file$_\n";
   exit(1);
  }
  if (!$foundManager && ($wantPMan ne "")){
    print "ERROR: manager '$wantPMan' for group '$wantPGroup' does not exist ";
    print "in your config file$_\n";
   exit(1);
  }
}

sub commandPGroups {
  my $self = shift;
  my ($command) = @_;
  my @pGroup = @{$self->{PGroups}};
 
  # Stop GLOBAL last, after others
  if (($command eq "STOP") || ($command eq "KILL")){

    foreach (@pGroup){ $_->command("STOP"); }
    ($self->{gPGroup})->command("STOP");

  # Start GLOBAL first, then others
  }elsif ($command eq "START"){

    ($self->{gPGroup})->command("START");
    foreach (@pGroup){ $_->command("START"); }

  # Stop GLOBAL between start and stop
  }elsif (($command eq "RESTART") || ($command eq "RELOAD")){
    foreach (@pGroup){ $_->command("STOP"); }
    ($self->{gPGroup})->command("STOP");
    print "Waiting 2 seconds for all processes to die...\n";
    sleep(2);
    ($self->{gPGroup})->command("START");
    foreach (@pGroup){ $_->command("START"); }

  # Heartbeat restart called from heartbeat script.  Hack to 
  # conditionally restart a possibly hung process
  # For now, ignore global
  }elsif ($command eq "HEARTBEAT"){
    foreach (@pGroup){ $_->command("HEARTBEAT"); }

  # Let group handle other commands directly
  # letting global go first
  # Rotate/Clear/Loginfo/Status
  }else{
    ($self->{gPGroup})->command($command);
    foreach (@pGroup){ $_->command($command); }
  }
}

sub updatePGroups {
  my $self = shift;
  my ($what, $wantPGroup, $wantPMan) = @_;

  # Load the required groups first
  $self->loadPGroups($wantPGroup, $wantPMan);

  # Now send them the command
  $self->commandPGroups($what);
}

#--------------------------------------------------------------------------
# setEnv -- Default ENV, which is anything in the pGroup GLOBAL section
#        -- No need for restore routine.  Only one PControl object
sub setEnv {
 my $self = shift;
 my $global = $self->{gPGroup};
 if ($global){ $global->setEnv(); }else{
   print "Warning...no global group environment defined\n";
 }
}

#--------------------------------------------------------------------------
# configFile
#
# -- Find our config file in various paths
sub configFile {
  my $self = shift;

  # Paths to check, in order of pref
  my @paths = ($ENV{HOME}, $RealBin, "/etc" );
  foreach (@paths){
    $toCheck = "$_/$PControl::CONFIG";
    if (-e $toCheck){ return $toCheck; }
  }

  # No configuration file found.  Die.
  print "ERROR:  Config file not found\n";
  foreach (@paths){
    print "Checked $_/$PControl::CONFIG\n";
  }
  exit(1);
}

#--------------------------------------------------------------------------
# readConfig
#
# Read in configuration file.  We look only for pGroups we need to run,
# ignoring everything else
#
sub readConfig {
  my $self = shift;
  my (@pGroups, $line);

  if (open(CONFIG, $self->configFile())){
    while($line = <CONFIG>){ 
      chop $line;
      if ($line =~ /^[ \t]*group ([^ \t]*)/i) { 
        push @pGroups, $1;
        next; 
      }
    }
  }
  close CONFIG;
  return @pGroups;
}
1;
