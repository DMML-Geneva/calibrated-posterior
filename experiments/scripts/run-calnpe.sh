#!/bin/bash
# Assuming the appropriate environment is loaded

cd workflows/calnpe || exit

if [[ "${#}" == 1 ]]; then
	PROB="${1}"
  echo " > Verifying ${PROB} pipeline."
  PYTHONPATH='../../:.' python pipeline.py --problem $PROB $DEBUG_FLAG
else
  rm -rf .workflows
  echo ' > Verifying Gravitational Waves pipeline.'
  PYTHONPATH='../../:.' python pipeline.py --problem gw $DEBUG_FLAG

  echo ' > Verifying Lotka Volterra pipeline.'
  PYTHONPATH='../../:.' python pipeline.py --problem lotka_volterra $DEBUG_FLAG

  echo ' > Verifying MG1 pipeline'
  PYTHONPATH='../../:.' python pipeline.py --problem mg1 $DEBUG_FLAG

  echo ' > Verifying SLCP pipeline'
  PYTHONPATH='../../:.' python pipeline.py --problem slcp $DEBUG_FLAG

  echo ' > Verifying Spatial SIR pipeline'
  PYTHONPATH='../../:.' python pipeline.py --problem spatialsir $DEBUG_FLAG

  echo ' > Verifying Weinberg pipeline'
  PYTHONPATH='../../:.' python pipeline.py --problem weinberg $DEBUG_FLAG
fi

cd ../..
