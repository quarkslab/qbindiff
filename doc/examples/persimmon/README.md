# Description
The purpose of this tutorial is to show how to perform an automated analysis on two version of a specified binary retrieved from persimmon and generate a report.

## Extract two binaries from persimmon
First we need to extract the two binaries from persimmon. We can use the python script `extract.py` to do it.

At the begin of the file we can configure some attributes:

```python
BASE_URL='http://kaki.persimmon.qb:8080/v0'
ARCH = 'x64'
KB_TARGET = '5013941'
BINARY = 'ntoskrnl.exe'
```

After having specified the right KB, arch and binary we can run the script that will download the aforementioned binary (and pdb) and its previous version (relative to the previous KB)

```commandline
python extract.py
```

Now we can find the binaries in the folder `output`

## Loading IDA

Now that we have the binaries we need to analyze them with IDA and export the result of the analysis with Quokka (or BinExport)

Note that you could use another tool to perform the analysis but you would need to either use BinExport or develop your own backend loader for qbindiff.

You can manually do it or use the script `analyze.py`

After that we should have the `.quokka` exported files.


## Perform the diffing

Now it's time to use QBinDiff to compute the real diff.

You can either use it with through the command line and generate a BinDiff file (aka sqlite database) or with the script `differ.py` that will output the result to stdout.
The diffing result will be the same in both cases.

### Command Line

```commandline
qbindiff -l quokka -e1 ./output/5012647-ntoskrnl.exe -e2 ./output/5013941-ntoskrnl.exe ./output/5012647-ntoskrnl.quokka ./output/5013941-ntoskrnl.quokka \
         -f wlgk:cosine -fopt wlgk max_passes 1 \
         -f fname:3 \
         -f dat \
         -f cst \
         -f addr:0.01 \
         -d canberra -s 0.999 -sr \
         -ff bindiff -o ./result.BinDiff -vv
```

Here I am limiting the number of passes of the Weisfeiler-Lehman Graph Kernel feature to 1 because otherwise it would be very slow but the number can be increased to achieve better accuracy.


### Using `differ.py`

```commandline
>> python differ.py


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
