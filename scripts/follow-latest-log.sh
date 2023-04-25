#!/bin/bash

FOLLOWDIR=${PWD}
FOLLOWLATEST=$(ls -1 -t ${FOLLOWDIR} | grep '^[a-zA-Z\-]*-[0-9\_]*\.out' | head -n 1)
FOLLOWLOG=${FOLLOWDIR}/${FOLLOWLATEST}
squeue

echo
echo "########################################################"
echo "Slurm job"
echo "Most recent job log: ${FOLLOWLATEST}"
echo "Following log: ${FOLLOWLOG}"
echo "########################################################"
echo

if [ "${FOLLOWLATEST}" == "" ]; then
	echo "No log files found"
	exit 0
fi

tail -n +0 -f ${FOLLOWLOG}
