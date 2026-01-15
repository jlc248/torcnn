#--------------------------------------------------------------------------
# Process Manager (PMan)
#
# REST Apr 2002
#
# -- Manager for a list of N unix processes
#

package PMan;
use PControl;
use PGroup;
use PLogUtil;
use Cwd;

$PControl::LASTRUNTIME    = "0";         # Last run time  

#--------------------------------------------------------------------------
# new ($pManName, $pGroupOwnerObject)
#
sub new
{
   my ($classname, $owner, $name) = @_;

   # This is the name we will refer to this instance of manager
   my ($sname);
   if ($name =~ /::(.*)/){ $sname = $1;}
   else {$sname = $name;}
   my ($self) = {
     'name'  => $sname,
     'cname' => $classname,
     'owner' => $owner,
     'ids'  => {},
     'env' => {},
     'renv' => {},
   };
   bless $self, $classname;
   $self;
}

#--------------------------------------------------------------------------
# command($what)
#
# Handle a command from above
#
sub command {
  my $self = shift;
  my ($what) = @_;

  $self->setEnv();
    if ($what eq "START"){
    $self->start();
    $self->loginfo();
    }
    $self->heartbeat() if ($what eq "HEARTBEAT");
    $self->stop() if ($what eq "STOP");
    $self->rotate() if ($what eq "ROTATE");
    $self->clear() if ($what eq "CLEAR");
    $self->loginfo() if ($what eq "LOGINFO");
    $self->statusForever() if ($what eq "STATUS");
    $self->tail() if ($what eq "TAIL");
  $self->restoreEnv();
}
sub start { my $self = shift; print "Manager started\n"; }

#--------------------------------------------------------------------------
# setIDS
#   -- subclass should set ids hash for each process id started
#
sub setID{
  my $self = shift;
  my ($key) = @_;
  ${$self->{ids}}{$key} = "1";
}

sub setIDS {
  my $self = shift; 
  print "ERROR::OBJECT.pm should have method setIDS defined!!!\n"; 
}

#--------------------------------------------------------------------------
# stop
#  -- Default method stops all processes we have a id for
#
sub stop { 
  my $self = shift;
  my $name = $self->{cname};

  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
   #print "$name --> Stopping process $_ \n";
   $self->stopForever("\U$_");
  }
}

#--------------------------------------------------------------------------
# heartbeat
#  -- Default method for a possibly stalled process
#
sub heartbeat { 
  my $self = shift;
  my $name = $self->{cname};

  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
    $ident = $self->getIdentifier($_);
    print "Heartbeat called for $ident\n";
    if ($self->isRunning($_)){
      print "-->Currently running, attempting restart\n";
      $self->stop();
      if (!($self->isRunning($_))){
        print "...successfully terminated process\n";
        sleep(2);
        $self->start();
      }else{
        print "...critical could not kill process!\n";
      }
    }else{
      print "-->process already stopped or dead.\n";
    }
  }
}

#--------------------------------------------------------------------------
# rotateClear
# -- routine shared by rotate and clear to do actual work
#  $what eq "clear" --> clear logs.  "rotate"|"" --> rotate
sub rotateClear {
  my $self = shift;
  my ($what) = @_;
  my ($logFile, $logNum, $logSize, $ident);
  my $name = $self->{name};

  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
   $ident = $self->getIdentifier($_);
   PLogUtil::getLogInfo("$ident#.log", "", \$logFile, \$logNum, \$logSize);

   if ($what eq "clear"){
     print "Clearing logs for $ident ---> $logFile\n";
     PLogUtil::clearLogs($logFile, $logNum);
   }else{
     print "Rotating logs for $ident ---> $logFile\n";
     PLogUtil::rotateLogs($logFile, $logNum);
   }
  }
}

sub rotate { my $self = shift; $self->rotateClear("rotate"); }
sub clear { my $self = shift; $self->rotateClear("clear"); }

#--------------------------------------------------------------------------
# loginfo
# -- Dump the log we are using to screen for each process
sub loginfo {
  my $self = shift;
  my ($logDir, $ident);

  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
    $ident = $self->getIdentifier($_);
    PLogUtil::getLogDir("", \$logDir);
    print "$ident\n --logging to---> $logDir/${ident}1.log\n";
  }
}

#--------------------------------------------------------------------------
# tail 
# -- Spawn a tail process for our first current log
#
sub tail {
  my $self = shift;
  my ($logDir, $ident);
  my $tail = $PControl::TAIL;

  # Humm.. tail first id found.  +++ probably have to redesign this later
  # maybe should open a xterm and put tail in it?
  # System will allow cntl-c to skip to next log.  Kinda cool
  $self->setIDS;
  foreach(keys %{$self->{ids}}){ 
    $ident = $self->getIdentifier($_);
    PLogUtil::getLogDir("", \$logDir);
    print "$ident\n --tailing log---> $logDir/${ident}1.log\n";
    if ($self->isRunning($_)){
    }else{
      print "**** Tailing log of process NOT CURRENTLY RUNNING  ****\n";
    }
    system "$tail $logDir/${ident}1.log";
  }
}

#--------------------------------------------------------------------------
# listParams
#
# --General methods for listing 'set' variables for this module
# Subclasses should override
sub listParams 
{
  my $self = shift;

  print "No parameters have been defined\n";
}

#--------------------------------------------------------------------------
# getIdentifier
#
#  Returns unique Identifier such as "KTLX-PROCESS-TOP"
#  Default is  groupname-managername-processid
sub getIdentifier {
  my $self = shift;
  my ($pID) = @_;
  my ($pGroup, $pMan);

  $pGroup = ($self->{owner})->getName();
  $pMan = $self->{name};
  return "\U$pGroup-$pMan-$pID";
}

#--------------------------------------------------------------------------
# --PS running  routines--
# runForever(ID, Args, [directory]); 
# stopForever(ID);
# statusForever(); -- print status for ALL ids
# isRunning(ID);
#  
# ID is an id for this process, will be put on the identifier
# it is used to uniquely define a process (for this PMan)
#
# allArgs -- command line to exec
# directory is optional home directory for process to be run from
#
#--------------------------------------------------------------------------
sub runForever
{
  my $self = shift;
  my ($ID, $allArgs, $dir) = @_;
  my $name = $self->{name};
  my ($ident) = $self->getIdentifier($ID);

  # Pause for at least wait X seconds between runForevers 
  my $curSeconds = `date +%s`;
  my ($timeSinceLast) = $curSeconds - $PControl::LASTRUNTIME;  
  my ($minTimeBetween) = $ENV{PROCESS_DWELL};
  if ($timeSinceLast < $minTimeBetween){ 
    my ($sleeper) = $minTimeBetween-$timeSinceLast;
    print "Waiting $sleeper (seconds) to keep  $minTimeBetween (seconds) between ";
    print "processes (set PROCESS_DWELL in conf file)\n";
    # See Programming Perl 'select' function.  Allows finer resolution sleep
    # For instance, user sets value to .5
    #sleep($sleeper);
    select undef, undef, undef, $sleeper;
  }

  # Here we make sure we aren't already running the process
  if ($self->isRunning($ID)){
    print "Already running $ident\n";
  }else{
    my ($exe) = $PControl::RUN;
    my ($logger) = $PControl::LOGGER;
    my ($procwatch) = $PControl::PROCER;
    my ($log) = $PControl::LOG;

    # Get rid of command line, used within w2run
    $allArgs =~ s/\\"/"/g;   # Change \" in file to real "
    $ENV{W2RUN_WHAT} = "$allArgs";
    $ENV{W2RUN_LOGGER} = "$logger";

    print "forever-->$name $allArgs\n";

    my $cmdLine = "$exe $ident | $logger $log w2 &";

    # Clean up last id before spawning
    my $lastid = $PControl::LASTRUN;
    unlink($lastid);

    # Change into requested directory before running (and then back)
    if (defined($dir)){ 
      my $curDir = cwd();
      chdir ($dir) or die "ERROR: Can't cd into $dir $!\n";
      system $cmdLine;
      chdir ($curDir) or die "ERROR: Can't restore $curDir $!\n";
    }else{
      system $cmdLine;
    }

    # Lovely, the last file hack is in a different thread,
    # so we have to wait forever for it pretty much
    # This could 'lock' if w2run fails to write id file
    # Only if we want procwatch...
    if ($ENV{PROC_WATCH_USE} eq "1"){
      my $notdone = 1;
      while ($notdone){
        if (open (LASTID, "<". "$lastid")){
           my $line = <LASTID>; chomp $line; close LASTID;
           if ($line =~ /[0-9][0-9]*/){
             $notdone = 0;
         
             my $watchDelay = $ENV{PROC_WATCH_SLEEP};
             if ($watchDelay < 2){ $watchDelay = 2; }
             # Spawn the w2procwatcher for this process...
             print "--->Spawning Proc Watcher for $line into log $ident#.proc\n"; 
             my $cmdLine = "$procwatch $line $watchDelay | $logger $ident#.proc w2 &";
             print "$cmdLine\n";
             system $cmdLine;
           }else{
             close LASTID; # Gotta go again, file there but ID not yet
           }
        }
      }
    } # End proc watcher start

    unlink($lastid);
  }

  # Get time AFTER sleep and system call
  $PControl::LASTRUNTIME = `date +%s`;  
}

sub stopForever
{
  my $self = shift;
  my ($ID) = @_;
  my ($ident) = $self->getIdentifier($ID);

  if ($self->isRunning($ID)){
    my $name = $self->{name};
    my $group = ($self->{owner})->{name};
    my ($exe) = $PControl::RUN;
    my ($col,$r,$a,$p,$arg, $leftSide, $rightSide, $pid);

    # Match up found processes in system that are running
    $col = $ENV{COLUMNS}; $ENV{COLUMNS} = 500;
    if (open (PS_PIPE, "ps -A -o pid -o args |")){
      while ($line = <PS_PIPE>){
         $line =~ s/\/\//\//g; # make any double / into single (bug work around)
         if ($line =~ /(.*)$exe(.*)/){   # Pull out processes first
           $leftSide = $1; $rightSide = $2;
           if ($leftSide =~ /([0-9]+)/){ $pid = $1; }
           else {$pid = "UNKNOWN";}
           if ($rightSide =~ / $ident[ \t\n]*/){
             print "TERM SENT: $ident PID: $pid\n";
               system ("kill -TERM $pid");
           }
        }
      }
    }
  }else{
    print "$ident had already died or been stopped\n";
  }
}

sub statusForever
{
  my $self = shift;
  my ($col) = $ENV{COLUMNS};
  my ($exe) = $PControl::RUN;
  my (@idents, $line, $leftSide, $pid, %found);

  # Get all identifiers for our processes
  $self->setIDS;
  foreach(keys %{$self->{ids}}){ push @idents, $self->getIdentifier($_); }

  # Open up ps.  +++ might be better to have group or control
  # open this pipe and feed us lines.
  # Right now we multiple ps for each status call
  $ENV{COLUMNS} = 500;
  if (open (PS_PIPE, "ps -A -o pid -o args |")){
    while ($line = <PS_PIPE>){
      $line =~ s/\/\//\//g; # make any double / into single (bug work around)
      foreach $ident (@idents){ 
         if ($line =~ /(.*)$exe $ident(.*)/){   # Pull out processes first
            $leftSide = $1;
            if ($leftSide =~ /([0-9]+)/){ $pid = $1; } else {$pid = "UNKNOWN";}
            $found{$ident} = $pid;
         }
      }
    }
    close (PS_PIPE);
 
    # Print found/notfound and all children processes 
    foreach (@idents){
      print "$_ ==> ";
      if ($found{$_}){ 
        # Get children of process quick and dirty
        if (open (PS_PIPE, "pstree $pid -pn |")){
          while ($line = <PS_PIPE>){
            chop $line;
            $line =~ s/ //g;
            $line =~ s/[-+\`\|]/ /g;
            print " $line";
          }
          print "\n";
        }else{
          print "(running?)\n";
        }
      }else{
        print  "(not running)\n";
      }
    }
  }else{
    print "Failed PS pipe for process checking.  Fatal error\n";
    print "'ps -A -o pid -o args' needs to work on your system \n";
    exit(1);
  }
  $ENV{COLUMNS} = $col;
}

sub isRunning
{
  my $self = shift;
  my ($ID) = @_;
  my ($line);
  my ($foundProcess) = 0;
  my ($col) = $ENV{COLUMNS};
  my ($exe) = $PControl::RUN;
  my ($find) = $self->getIdentifier($ID);

  # Check each process...
  $ENV{COLUMNS} = 500;
  if (open (PS_PIPE, "ps -A -o args |")){
    while ($line = <PS_PIPE>){
      $line =~ s/\/\//\//g; # make any double / into single (bug work around)
      # we look for something like "w2alg ktlx-oclock-a"
      if ($line =~ /$exe $find/){ $foundProcess = 1; last; } 
    }
    close (PS_PIPE);
  }else{
    print "Failed PS pipe for process checking.  Fatal error\n";
    print "'ps -A -o args' needs to work on your system \n";
    exit(1);
  }
  $ENV{COLUMNS} = $col;
  return ($foundProcess);
}

#--------------------------------------------------------------------------
# --ENV routines--
#
# initEnv - initial set of env during load
# setEnv/restoreEnv -- setup our ENV for each process
# envOr -- return ENV value or default
#
sub initEnv
{
  my $self = shift;
  my ($var, $value) = @_;
  ${$self->{env}}{$var} = $value;
}
sub setEnv { 
  my $self = shift;

  # Now set our particular variables
  # note, these may override any PGroup variables
  $env = $self->{env};
  while(($key, $value) = each(%$env)){ 
    ${$self->{renv}}{$key} = $ENV{$key};   # Save to restore
    $ENV{$key} = $value; 
  }
}
sub restoreEnv {
  my $self = shift;

  $renv = $self->{renv};
  while(($key, $value) = each(%$renv)){ 
    $ENV{$key} = $value;
  }
}
sub envOr{
  my $self = shift;
  my ($envName, $default) = @_;
  #if (defined ($ENV{$envName})){ return $ENV{$envName}; 
  if (defined ($ENV{$envName})){
    my $value = $ENV{$envName};

    # Replace macros here.  Might be more efficient higher up
    # if too slow probably move up into PMan setEnv and PGroup setEnv
    my $i;
    for($i = 1; $i < 50; $i++){
      if (defined ($ENV{"M$i"})){
        my $macro = $ENV{"M$i"};
        $value =~ s/\$M$i([^0-9])/$macro$1/gi;
      }
    }

    return $value;
  }else{ return $default; }
}
1;
