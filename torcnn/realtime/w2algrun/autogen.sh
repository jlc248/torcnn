#!/bin/sh

# Extremely simple script to compile the two really used scripts
# Another option is to just cp w2logger.pl to w2logger, etc.

# Set up an initial w2alg.conf file if needed
if [ -f w2alg.conf ]; then
  echo "w2alg.conf exists..."
else
  echo "w2alg.conf does not exist, copying example into it"
  cp w2algEXAMPLE.conf w2alg.conf
fi

# Compile w2run and w2logger for speed
if false; then
echo "Compiling w2run..."
perlcc -o w2run w2run.pl
echo "Compiling w2logger..."
perlcc -o w2logger w2logger.pl
echo "Compiling w2procwatch..."
perlcc -o w2procwatch w2procwatch.pl
else
# Copy w2run.pl and w2logger.pl for small memory
# Doesn't seem to help speed much anyway
echo "Setting up w2run..."
cp w2run.pl w2run
echo "Setting up w2logger..."
cp w2logger.pl w2logger
echo "Setting up w2procwatch..."
cp w2procwatch.pl w2procwatch
fi


echo "You can make a symbolic link to w2alg in this directory from /usr/bin if you want:"
echo "Example: ln -s /home/you/w2algrun/w2alg /usr/bin/w2alg"
