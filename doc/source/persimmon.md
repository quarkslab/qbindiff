# PersimMon

```{note}
All the files that are nominated in this documentation can be found in the folder [`doc/examples/persimmon/`](https://gitlab.qb/machine_learning/qbindiff/-/tree/master/doc/examples/persimmon/)
```

## Description
This tutorial shows how to perform an automated diffing of two versions of a specific binary retrieved from PersimMon and generate a report.

## Prerequisites

The following requirements are needed for this tutorial:

* An access to PersimMon (ping @gwaby)
* A functional installation of QBinDiff (with Quokka / BinExport backends)
* A working version of a supported disassembler (IDA)

## Extract two binaries from PersimMon
First, we need to extract the two binaries from PersimMon. We can use the python script [`extract.py`](https://gitlab.qb/machine_learning/qbindiff/-/tree/master/doc/examples/persimmon/extract.py) to do it.

At the beginning of the file we can configure some attributes:

```python
BASE_URL = 'http://kaki.persimmon.qb:8080/v0'
ARCH = 'x64'
KB_TARGET = '5013941'
BINARY = 'ntoskrnl.exe'
```

After having specified the right KB, arch and binary running the script will download the aforementioned binary (and pdb) with its previous version (relative to the previous KB).

```bash
python extract.py
```

The binaries are downloaded in the folder `output`.

## IDA Disassembly

Now that we have the binaries we need to disassemble them with IDA and export the result of the analysis with Quokka (or BinExport)

```{note}
It is possible to use another disassembler (e.g. Ghidra or Binary Ninja) when exporting using BinExport.

It is also possible to write your own custom qbindiff backend loader that loads the disassembly analysis from your third party tool of choice. See {ref}`usage/custom_backend` for more information.
```

The script [`analyze.py`](https://gitlab.qb/machine_learning/qbindiff/-/tree/master/doc/examples/persimmon/analyze.py) exports the binaries in `output` with Quokka.
After that, we should have the `.quokka` exported files.

## Perform the diffing

Now it's time to use QBinDiff to compute the real diff.

You can either use it with the command line and generate a BinDiff file (aka sqlite database) or with the script [`differ.py`](https://gitlab.qb/machine_learning/qbindiff/-/tree/master/doc/examples/persimmon/differ.py) that will output the result to stdout.
The diffing result will be the same in both cases.

### Command Line

```bash
qbindiff -l quokka -e1 ./output/5012647-ntoskrnl.exe -e2 ./output/5013941-ntoskrnl.exe \
         ./output/5012647-ntoskrnl.quokka ./output/5013941-ntoskrnl.quokka \
         -f wlgk:cosine -fopt wlgk max_passes 1 \
         -f fname:3 \
         -f dat \
         -f cst \
         -f addr:0.01 \
         -d canberra -s 0.999 -sr \
         -ff bindiff -o ./result.BinDiff -vv
```

The previous command use a set of options that can scare an user. Let's detail them. For a more extensive description of all the command line arguments see [here](usage/qbindiff).

- `-l` (for loader) specifies that we are using a Quokka backend
- `-e1` is for the first sample **binary**
- `-e2` is for the second sample **binary**
- The next two arguments are the only mandatory positional arguments. They are the two **export** files
- Then we use the `-f` (for feature) option to list the features we want to use.
  - `wlgk` (for Weisfeiler-Lehman Graph Kernel) use the distance `cosine` and is modified by the `-fopt` flag to restrict the number of passes.
  - `fname`, uses the name of the function if the binary has not been stripped. It is specified with a weight of `3` (by default the weight is 1).
  - `dat`, matches the same data references
  - `cst`, same usage of constants, also known as immutables
  - `addr`, match two functions if they starts at the same address. It is not a meaningful feature, here it is used just to introduce a small noise (weight `0.01`) to let the algorithm to distinguish between multiple identical candidates
- `-d` (for distance) selects the default distance to use where not otherwise specified (here: `canberra`)
- `-s` defining the sparsity ratio of the similarity matrix. `0.999` means that the 99.9% of the matrix will be filled with zeros
- `-sr`, by specifying this flag we are saying that the density value of the sparse similarity matrix will affect each row the matrix. That means that each row of the matrix has the defined sparsity ratio. This guarantees that there won't be rows that are completely erased and filled with zeros.
- `-ff` (for file format) the output format
- `-o` (for output) the output file
- `-vv` to increase the verbosity

```{note}
`-e1` and -`e2` arguments are optional in general but they are mandatory if using quokka backend
```
```{note}
Here, we are limiting the number of passes of the Weisfeiler-Lehman Graph Kernel feature to 1 because otherwise it would be very slow but the number can be increased to achieve better accuracy.
```
```{note}
The less sparse the similarity matrix is the better accuracy we achieve but also the slower the algorithm becomes. In general the right value must be tuned for each use case.

Also note that after a certain threshold reducing the sparsity of the matrix **won't** yield better results.
```

### Using [`differ.py`](https://gitlab.qb/machine_learning/qbindiff/-/tree/master/doc/examples/persimmon/differ.py)

```bash
python differ.py


[...SNIP...]
=== REPORT ===
	Similarity: 0.9985966527410982
	21924 functions have not been modified
	22 functions have been modified
	23 functions have been deleted
	9 functions have been added
-- MODIFIED FUNCTIONS --
	PsGetCurrentServerSiloGlobals  ->  PsGetCurrentServerSiloGlobals  similarity: 0.1640625 confidence: 1.0
	PerfInfoLogSysCallEntry  ->  PerfInfoLogSysCallEntry  similarity: 0.0 confidence: 0.999919593334198
	PsIsCurrentThreadInServerSilo  ->  PsIsCurrentThreadInServerSilo  similarity: 0.0 confidence: 1.0
	NtQueryInformationProcess  ->  NtQueryInformationProcess  similarity: 0.3359375 confidence: 1.0
	MmIsSessionInCurrentServerSilo  ->  MmIsSessionInCurrentServerSilo  similarity: 0.71875 confidence: 0.9999864101409912
	MiGetNextSession  ->  MiGetNextSession  similarity: 0.78125 confidence: 0.9999936819076538
	IopCheckSessionDeviceAccess  ->  IopCheckSessionDeviceAccess  similarity: 0.515625 confidence: 0.9997091889381409
	RtlGetCurrentServiceSessionId  ->  RtlGetCurrentServiceSessionId  similarity: 0.5625 confidence: 0.9990647435188293
	MmGetSessionById  ->  MmGetSessionById  similarity: 0.5546875 confidence: 1.0
	EtwpTraceFileName  ->  EtwpTraceFileName  similarity: 0.453125 confidence: 0.9999531507492065
	sub_14066D1F0  ->  sub_14067A530  similarity: 0.0 confidence: 0.9932856559753418
	sub_14066BB10  ->  sub_140684030  similarity: 0.0 confidence: 0.9932856559753418
	RtlGetNtProductType  ->  RtlGetNtProductType  similarity: 0.265625 confidence: 0.9999973177909851
	PsGetCurrentServerSilo  ->  PsGetCurrentServerSilo  similarity: 0.28125 confidence: 1.0
	PerfInfoLogSysCallExit  ->  PerfInfoLogSysCallExit  similarity: 0.0 confidence: 0.999919593334198
	RtlGetActiveConsoleId  ->  RtlGetActiveConsoleId  similarity: 0.1484375 confidence: 0.9999988675117493
	PopPowerRequestCreateInfo  ->  PopPowerRequestCreateInfo  similarity: 0.375 confidence: 0.999935507774353
	PspRundownSingleProcess  ->  PspRundownSingleProcess  similarity: 0.4609375 confidence: 1.0
	MmSessionGetWin32Callouts  ->  MmSessionGetWin32Callouts  similarity: 0.5625 confidence: 0.999997079372406
	sub_14067B250  ->  sub_14066D540  similarity: 0.0 confidence: 0.9932856559753418
	NtSetInformationProcess  ->  NtSetInformationProcess  similarity: 0.3515625 confidence: 1.0
	SepAdtLogAuditRecord  ->  SepAdtLogAuditRecord  similarity: 0.421875 confidence: 1.0
-- DEAD FUNCTIONS --
	sub_140684560
	sub_140789445
	sub_1407894C3
	sub_140789559
	sub_14078964E
	sub_1407896C2
	sub_140789783
	sub_140789A5D
	sub_140789ADC
	sub_140789B87
	sub_140789C11
	sub_140789CCE
	sub_140789D00
	sub_14078ADB9
	sub_14078AFF2
	sub_14078B0B2
	sub_14078B175
	sub_14078C51D
	sub_14078C5B5
	sub_14078C699
	sub_14078C75B
	PspGetWakeCountProcess
	PspWriteProcessSecurityDomain
-- NEW FUNCTIONS --
	sub_14066C570
	IoCheckRedirectionTrustLevel
	IoComputeRedirectionTrustLevel
	KeIsExecutingInArbitraryThreadContext
	SeTokenGetRedirectionTrustPolicy
	SeTokenSetRedirectionTrustPolicy
	PspGetRedirectionTrustPolicy
	PspSetRedirectionTrustPolicy
	EtwTimLogRedirectionTrustPolicy
```

From this result, it is possible to load the database in BinDiff and continue to work on the results.

This wraps up our first tutorial. Good Reversing ;)
