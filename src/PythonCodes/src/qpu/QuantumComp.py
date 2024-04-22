#!/usr/bin/env python3
'''!\file
   -- DataManage addon: (Python3 code) class for handling some of the command line
      of the common variables
      \author Frederic Bonnet
      \date 27th of February 2024

      Universite de Perpignan March 2024

Name:
---
Command_line: class Usage_Network for extracting network transfers between
two interfaces.

Description of classes:
---
This class generates an object that contains and handlers

Requirements (system):
---
* sys
* psutil
* time
* os
* pandas
* datetime
'''
# System imports
import sys
import psutil
import time
import os
import pandas as pd
from datetime import datetime
import numpy as np

from qat.lang.AQASM import * #Program, X, H, CNOT, SWAP
from qat.qpus import get_default_qpu
import sympy

import cirq
# Path extension
sys.path.append(os.path.join(os.getcwd(), '.'))
sys.path.append(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(os.getcwd(), '..','..'))
#Application imports
#import src.PythonCodes.DataManage_common
# Global variables
UPDATE_DELAY = 1     # (secs)
# Definiiton of the constructor
class QuantumComp:
    # Constructor
    def __init__(self, c, m):
        __func__= sys._getframe().f_code.co_name
        self.rc = 0
        self.c = c
        self.m = m
        self.app_root = self.c.getApp_root()  #app_root
        self.m.printMesgStr("Instantiating the class : ", self.c.getGreen(), "QuantumComp")

# get_size(self, bytes)
    def BellState(self, qubit_count):
        rc = self.c.get_RC_SUCCESS()

        # Greeting message
        self.m.printMesg("Hello Bell State test...")

        # Get the cubit circuit
        qreg = [cirq.LineQubit(x) for x in range(2)]
        circuit = cirq.Circuit()

        # construct the circuit
        circuit.append([cirq.H(qreg[0]), cirq.CNOT(qreg[0], qreg[1])])
        # Print the circuit diagram
        self.m.printLine()
        print(circuit)

        # Add measrument
        circuit.append(cirq.measure(*qreg, key="z"))

        # Simulate the circuit
        sim = cirq.Simulator()
        res = sim.run(circuit, repetitions=100)

        # Diusplay the results
        print("Measurements:")
        print(res.histogram(key="z"))
        # Goodbye message
        self.m.printMesg("Bye Bell state")

        return rc
    # extract_speed(self, speed_string)
    def Circuits_pyAQASM(self, qubit_count):
        rc = self.c.get_RC_SUCCESS()
        # Greeting message
        self.m.printMesg("Hello pyAQASM test...")

        my_program = Program()

        qbits = my_program.qalloc(2)

        #my_program.apply(X, qubit_reg[0])
        my_program.apply(H, qbits[0])
        my_program.apply(CNOT, qbits[0], qbits[1])

        circuit = my_program.to_circ()

        #circuit.display()

        mon_job = circuit.to_job()

        linalagqpu = get_default_qpu()

        result = linalagqpu.submit(mon_job)

        for sample in result:
            print("State %s amplitude %s" % (sample.state, sample.amplitude))

        result.plot()

        job_10 = circuit.to_job(nbshots=10)
        results_10 = linalagqpu.submit(job_10)


        for sample in results_10:
            print("State %s amplitude %s" % (sample.state, sample.amplitude))

        return rc
    # get_MBytess(self, speed, units)
    def Quantum_FullAdder(self, qubit_count):
        rc = self.c.get_RC_SUCCESS()
        # Greeting message
        self.m.printMesg("Hello full_Adder...")

        qprog = Program()

        qbits = qprog.qalloc(qubit_count)

        qprog.apply(CCNOT, qbits[0],qbits[1], qbits[3])
        qprog.apply(CNOT, qbits[0], qbits[1])
        qprog.apply(CCNOT, qbits[1], qbits[2],qbits[3])
        qprog.apply(CNOT, qbits[1], qbits[2])
        qprog.apply(CNOT, qbits[0], qbits[1])

        full_adder = qprog.to_circ()

        #full_adder.display()

        # Carry input
        carry_input = Program()
        qbits = carry_input.qalloc(4)
        carry_input.apply(X, qbits[0])
        carry_input.apply(X, qbits[1])
        carry_input.apply(X, qbits[2])

        input_circ = carry_input.to_circ()

        input_circ.display()

        full_circuit = input_circ + full_adder
        full_circuit.display()

        # Now the addition
        job = full_circuit.to_job()
        qpu = get_default_qpu()
        result = qpu.submit(job)
        for sample in result:
            print("State %s: Nb %s, probability %s, amplitude %s" %
                  (sample.state, sample.state.int, sample.probability, sample.amplitude))

        # Carry input
        input = Program()
        qbits = input.qalloc(4)
        input.apply(H, qbits[0])
        input.apply(H, qbits[1])
        input.apply(H, qbits[2])

        input_cirq = input.to_circ()

        input_circ.display()

        return rc

    def quantum_or(self, qubit_count):
        rc = self.c.get_RC_SUCCESS()
        self.m.printMesg("Hello quantum_or...")
        or_program = Program()
        qbits = or_program.qalloc(qubit_count)

        or_program.apply(X, qbits[0])
        or_program.apply(X, qbits[1])
        or_program.apply(CCNOT, qbits[0],qbits[1], qbits[2])
        or_program.apply(X, qbits[0])
        or_program.apply(X, qbits[1])
        or_program.apply(X, qbits[2])



        or_circuit = or_program.to_circ()
        or_circuit.display()


        return rc

    # Orcale
    def by_oracle(self, secret):
        rc = self.c.get_RC_SUCCESS()
        n = len(secret)
        oracle = Program()
        qbits = oracle.qalloc(n+1)
        for i,query in enumerate(secret):
            if query == "1":
                CNOT(qbits[i], qbits[n])
        circ_oracle = oracle.to_circ()

        return circ_oracle
    def Bernstein_Vazirani(self, secret, qubit_count):
        rc = self.c.get_RC_SUCCESS()

        before_oracle = Program()
        qbits = before_oracle.qalloc(qubit_count+1)
        before_oracle.apply(X, qbits[qubit_count])
        for qbit in qbits[0:qubit_count+1]:
            before_oracle.apply(H, qbit)

        oracle = self.by_oracle(secret)

        after_oracle = Program()
        qbits = after_oracle.qalloc(qubit_count+1)
        for qbit in qbits[0:qubit_count+1]:
            after_oracle.apply(H, qbit)

        # Building the circuit
        circuit = before_oracle.to_circ() + oracle + after_oracle.to_circ()
        circuit.display()

        job = circuit.to_job()

        result = get_default_qpu().submit(job)
        for sample in result:
            print("State %s: Nb %s, probability %s, amplitude %s" %
                  (sample.state, sample.state.int, sample.probability, sample.amplitude))









        return rc
    def Usage_NetworkInterfaces_onePass(self, log_file):
        rc = self.c.get_RC_SUCCESS()
        return rc
#---------------------------------------------------------------------------
# end of Usage_Network
#---------------------------------------------------------------------------
