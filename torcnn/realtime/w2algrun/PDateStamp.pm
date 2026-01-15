#!/usr/bin/perl -w

package PDateStamp;

sub dateStamp{
 #my $date = `date -u`;
 #chop $date;

  # Use builtin perl function for date to help portability
  my @u = gmtime(time);
  my $mon = (qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec))[$u[4]];
  my $year = ($u[5]+1900);
  my $day = (qw(Sun Mon Tue Wed Thu Fri Sat))[$u[6]];
  my $num = $u[3];
  my $sec = $u[0]; if ($sec < 10) { $sec = "0$sec"; }
  my $min = $u[1]; if ($min < 10) { $min = "0$min"; }
  my $hour = $u[2]; if ($hour < 10) { $hour = "0$hour"; }

  return sprintf "[%3s %3s %2s %2s:%2s:%2s UTC %4s] ",
   $day, $mon, $num, $hour, $min, $sec, $year;
  #return "[$date] ";
}
1;
