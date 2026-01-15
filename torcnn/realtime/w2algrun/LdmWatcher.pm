#!/usr/bin/perl -w
#--------------------------------------------------------------------------
#
# REST 2003
#
# LdmWatcher--Class for watching ldm files in a directory.
# Checks file name to try to find bad ones.
#--------------------------------------------------------------------------
package LdmWatcher;

#
# Watch a directory for changes
#
use PDirwatch;
@ISA = qw(PDirwatch);

sub new() 
{
  my ($type) = shift;
  my ($self) = {};
  $self->{BADFILE} = "Nobadfile";
  bless ($self, $type);
  return $self;
}

sub checkFileIntegrity($)
{
  my ($self) = shift;
  my ($fileToCheck, $timer, $dir) = @_;
  # Do nothing...subclass can check file integrity

  # OKC20030419-152881-1.00-23-260-0.bz2
  if ($fileToCheck =~ /^([^-]*)-([^-]*)-([^-]*)-([^-]*)-([^-]*)/){
    if ($fileToCheck eq $self->{BADFILE}){
      # File same as before...ignore
    }else{
      #$self->myprint("File number is $1 $2 $3 $4 $5 \n");

      # Condition to cause email.  In this case radial less than this number
      if ($5 < 320){
         $self->{BADFILE} = $fileToCheck;
         $self->fireBadFileEmail($fileToCheck, $timer, $dir);
      }
    }
  }else{
      # Ignore if not a match.  Have no idea then...
  }
}

sub fireBadFileEmail()
{
  my ($self) = shift;
  my ($filename, $timer,$dir) = @_;
  my ($message);
  my ($date) = `date`;
  my ($machine) = `hostname`;

  $message = "'$machine' reporting, local date $date\n";
  $message = "Possible corrupted file detected $filename\n";
  $self->fireEmail($timer, $dir, $message);
}
