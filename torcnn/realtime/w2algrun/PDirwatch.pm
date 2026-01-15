#!/usr/bin/perl -w
#--------------------------------------------------------------------------
#
# REST 2003
#
# PDirwatch --Class for watching files in a directory
#--------------------------------------------------------------------------
package PDirwatch;

#
# Watch a directory for changes
#
use strict;
use PDateStamp;

# Globals to save run time
my ($perlFileFilter) = "";
my (@simpleFilterList) = "";
my ($outputdir) = "/tmp";

sub new()
{
   my ($type) = @_;
   my ($self) = { };
   bless $self, $type;
   return $self;
}

# Regular print interferes with the perl wrapper...probably changing
# variables.  Humm..not good
sub myprint()
{
  my ($self) = shift;
  my ($toPrint) = @_;
  chomp $toPrint;
  system("echo \"$toPrint\"");
}

# 'live' step-by-step output.  At least for debugging
sub liveprint()
{
  my ($self) = shift;
if (0){
  my ($toPrint) = @_;
  chomp $toPrint;
  system("echo \"$toPrint\"");
}
}

sub careAboutFile()
{
  my ($self) = shift;
  my ($file) = @_;
  my $curFilter;

  # ignore . and .. always
  if ($file =~ /^\.\.?$/){ return 0; }

  # Perl filter.  Very sensitive
  if ($perlFileFilter ne ""){
    if ($file =~ /$perlFileFilter/){ return 1; }
  }
  foreach $curFilter (@simpleFilterList){
    if ("\U$file" =~ /$curFilter/){ return 1; }  # allow these
  }
  return 0;                                  # default is ignore
}

sub getLatestDirTime()
{
  my ($self) = shift;
  my ($dir, $latestFilename) = @_;

  # Get newest file in directory.  This is O(N).. easier way?
  # Tempted to use ls -lt instead, parse top line
  # might do that if the directory size starts lagging us
  $$latestFilename = "";
  opendir(DIR, $dir) || $self->myprint("open dir $dir failed\n");
  my $latestWritetime = 0;
  my ($file, $readtime, $writetime);
  while(defined($file = readdir(DIR))){
     next if $file =~ /^\.\.?$/; # skip . and ..

     # Only check files matching one of our filters...
     if ($self->careAboutFile($file)){
       ($readtime, $writetime) = (stat("$dir/$file"))[8,9];
       #$self->myprint("$file written at $writetime");
       if ($writetime > $latestWritetime){ 
          $latestWritetime = $writetime; 
          $$latestFilename = $file;
       }
     }
  }
  closedir(DIR);
  return $latestWritetime;
}

# This will be an attempt to keep statistics on files...
# Average time, etc...
sub writeAverage()
{
  my ($self) = shift;
  my ($waited) = @_;
  my ($file) = $outputdir."/average.txt";
  if (open(DLOGFILE, ">".$file)){
    print DLOGFILE PDateStamp::dateStamp();
    print DLOGFILE $waited;
    print DLOGFILE "\n";
  }
  close DLOGFILE;
}

sub writeOverdueLog()
{
  my ($self) = shift;
  # Output gap in file data 
  my ($last, $now, $totalTime) = @_;
  my $out = "$last <--> $now ($totalTime sec gap)";
  $self->myprint($out);
}


sub fireEmail()
{
  my ($self) = shift;
  my ($timer, $dir, $message) = @_;
  my ($mail) = "";
  my ($sender, $receiver);

  if (defined ($ENV{"DIRWATCH_EMAILS"}) && 
      defined ($ENV{"DIRWATCH_EMAILR"}))
  {
     $sender = $ENV{"DIRWATCH_EMAILS"};
     $receiver = $ENV{"DIRWATCH_EMAILR"};
  }else{
     $self->myprint("-->Email not sent.  Define EMAILS and EMAILR in config file\n");
     return;
  }
  
  $mail=<<ENDMAIL;
From: $sender
To: $receiver
Subject: Directory Watcher Notification
Mime-Version: 1.0
Content-Type: text/html

$message
ENDMAIL
  # Use send mail for message...
$self->myprint($mail);
  if (open(M, "| /usr/sbin/sendmail -t -oi -f\"$sender\"")){
    print M $mail;
    close M
  }
  $self->myprint("-->Email sent at $timer secs\n");
}

sub fireOverdue()
{
  my ($self) = shift;
  my ($lastFile, $timer,$dir) = @_;
  my ($message);
  my ($date) = `date`;
  my ($machine) = `hostname`;

  $message = "'$machine' reporting, local date $date\n";
  $message .= "Data appears to be lagging for directory: $dir\n";
  $message .= "$timer secs without update event.\n";
  $message .= "Last file seen is $lastFile\n";
  $message .= "I'll send you another email when it comes back up.\n";
  $self->fireEmail($timer, $dir, $message);
}

sub fireOverdueRestored()
{
  my ($self) = shift;
  my ($lastFile, $newestFile, $timer,$dir) = @_;
  my ($message);
  my ($date) = `date`;
  my ($machine) = `hostname`;

  $message = "'$machine' reporting, local date $date\n";
  $message = "Data appears to be restored for directory $dir\n";
  $message .= "$timer secs before update event.\n";
  $message .= "File gap is $lastFile <---> $newestFile\n";
  $message .= "I'm back to normal watching...\n";
  $self->fireEmail($timer, $dir, $message);
}

sub checkFileIntegrity()
{
  my ($self) = shift;
  my ($fileToCheck, $timer, $dir) = @_;
  # Do nothing...subclass can check file integrity
  $self->myprint("Here in root class $fileToCheck\n");
}

sub mainWatchLoop {
  my ($self) = shift;
  my ($dir);
  my ($oldTime);
  my ($latestTime);
  my ($totalSeconds);       # Seconds passed without a verifed new/changed file
  my ($waitSeconds) = 5;    # Seconds to wait between checking for 
                            # new/changed files
  my ($waitMax) = 6000;     # Total seconds to wait before logging
  my ($waitCMax) = 60000;   # Total seconds to wait before emailing
  my ($overdue) = 0;        # Is a change overdue?
  my ($coverdue) = 0;       # Is a change Critically overdue (need an email)?
  my ($newestFile) = "";
  my ($lastFile) = "None";

  # We don't check the validness of any param, that's the DIRWATCH.PM's job
  
  # Directory we watch...
  if (defined ($ENV{"DIRWATCH_WATCH_DIR"})){
     $dir = $ENV{"DIRWATCH_WATCH_DIR"};
  }else{ $dir = ""; }

  # Read in the simple filter list, if any
  # Format is "*.tar.gz, *.zip " etc...
  if (defined ($ENV{"DIRWATCH_FILTER"})){
    my $temp = $ENV{"DIRWATCH_FILTER"};
    $temp =~ s/[ \t\\]//g;  # get rid of tabs, spaces, \
    $temp =~ s/[\.]/\\./g;  # . --> \.
    $temp =~ s/[\*]/\.*/g;  # * --> .*
    $temp = "\U$temp";      # Upper case match
    @simpleFilterList = split /,/, $temp;
  }

  # Read in the perl filter, if any
  if (defined ($ENV{"DIRWATCH_PERL_FILTER"})){
     $perlFileFilter = $ENV{"DIRWATCH_PERL_FILTER"};
  } 
 
  # Check time
  if (defined ($ENV{"DIRWATCH_CHECK_TIME"})){
     $waitSeconds = $ENV{"DIRWATCH_CHECK_TIME"};
  } 

  # Complain time
  if (defined ($ENV{"DIRWATCH_OVERDUE_TIME"})){
     $waitMax = $ENV{"DIRWATCH_OVERDUE_TIME"};
  } 

  # Critical complain time
  if (defined ($ENV{"DIRWATCH_EOVERDUE_TIME"})){
     $waitCMax = $ENV{"DIRWATCH_EOVERDUE_TIME"};
  } 

  # Directory we place our logs/html, etc...
  if (defined ($ENV{"DIRWATCH_LOG_DIR"})){
     $outputdir = $ENV{"DIRWATCH_LOG_DIR"};
  } 

  # Infinite loop until we are force stopped.. +++ catch signal
  $self->liveprint("Watching directory $dir, checking every ($waitSeconds) secs\n");
  $oldTime = $self->getLatestDirTime($dir, \$newestFile);
  $totalSeconds = 0;
  $overdue = 0;
  $coverdue = 0;
  while(1){
    sleep($waitSeconds);   # Time between checks...

    # Time changed, either old file modified or new one made...
    $lastFile = $newestFile;
    $latestTime = $self->getLatestDirTime($dir, \$newestFile);
    $self->checkFileIntegrity($newestFile, $totalSeconds, $dir);

    if ($latestTime > $oldTime){           # Changed/added file
       $self->liveprint(
"New/Change? LatestTime: $latestTime  Waited:$totalSeconds sec(s) $newestFile\n");
       $self->writeAverage($totalSeconds);
       if ($overdue){
         $self->liveprint("--was overdue by approximately $totalSeconds sec(s)\n");
         $self->writeOverdueLog($lastFile, $newestFile, $totalSeconds);
       }
       if ($coverdue){
         $self->liveprint("Sending data restored email at $totalSeconds sec(s)\n");
         $self->fireOverdueRestored($lastFile, $newestFile, $totalSeconds, $dir);
       }
       $oldTime = $latestTime;
       $totalSeconds = 0;
       $overdue = 0;
       $coverdue = 0;
    }elsif ($latestTime < $oldTime){       # File deleted probably
       $self->liveprint(
"Deletion of latest file?\n LatestTime decreased: $latestTime  Waited:$totalSeconds sec(s) $newestFile\n");
       $oldTime = $latestTime;
       #$totalSeconds = 0;  doesn't count as restart (should be option)
    }else{
       $totalSeconds += $waitSeconds;      # Just keep twiddling our thumbs
    }

    # If we haven't had a file written or changed within wait max, then
    # we need to remember this...
    if ($totalSeconds >= $waitMax){
       if ($overdue == 0){
         $self->liveprint("New/Changed overdue by $totalSeconds sec(s)\n");
         $overdue = 1;
       }
    }
    # Critical max check
    if ($totalSeconds >= $waitCMax){
       if ($coverdue == 0){
         $self->liveprint("New/Changed VERY overdue by $totalSeconds sec(s)\n");
         $coverdue = 1;
         $self->fireOverdue($lastFile, $waitCMax, $dir);
       }
    }
  }
}
