# Vendor claims vs. our measurements

> **Canonical copy:** this document now lives in the `cs3-microbench` repo (`docs/vendor-claims-vs-measurements.md`); this copy is kept because the ADRs cite it and is not updated further.

What Cerebras documents about host↔wafer transfer on the CS-3, next to what we
measured through the public SDK from an external user account. Quotes are
verbatim, with the source and the date they were checked. Every measurement row
names its commit and results CSV.

**Status of our numbers.** Streaming times recorded before `080e3ca` are
*host-return* times. They are lower bounds on the true transfer time, so any
rate computed from them is an upper bound (see `docs/host-timing-barrier.md`).
Rows below say which basis they use. **Verified** means measured with the completion barrier on 2026-10-07 (`cluster/results/data/2026-10-07_barrier-remeasure.csv`, 65,536 PEs, 114,000² at d = 1e-3, on one system). Today's runs are slower than identical runs on 3–6 Oct even in host-return time, so compare verified values only with each other.

Sources checked 2026-10-07:
- [SDK] Host Runtime and Tensor Streaming, https://sdk.cerebras.ai/tensor-streaming
- [EX] `csl-examples/benchmarks/bandwidth-test/run.appliance.py` (the vendor's own bandwidth benchmark)
- [CLI] `cerebras/sdk/client/sdk_appliance_utils.py`, cerebras-sdk 2.10.0 on the EPCC user node (the `--channels`, `--width-*-buf` help text)
- [SYS] Cerebras CS-3 system page / spec sheet, as summarised by third parties (the vendor page now shows CS-4; to be pinned against a CS-3 datasheet PDF)
- [TICKET] Support-ticket reply (not public, not citable as is)

## 1. Memory order

| | |
|---|---|
| **Vendor** | [SDK] "Column-major delivers better bandwidth than row-major." |
| **Measured, copy mode** | Holds. 65,536 PEs, 16 channels, 302.0 MB setup transfer: row-major 7.23 / 7.12 GB/s, col-major **10.35 GB/s (+43%)**. Commit `cd6ee36`. Basis: host-return of a blocking copy. **Verified: col-major setup 44.7–47.3 ms = 6.39–6.75 GB/s, against row-major 58.0–60.6 ms = 4.99–5.20 GB/s, so +30%** (16 channels, 302 MB). |
| **Measured, streaming** | Doesn't hold for our per-PE streaming kernel. Host-return times: col-major 43.5–44.0 ms vs row-major 20.0–20.3 ms. **Verified at 16 channels, full mode: col-major 244.8 ms vs row-major 85.9–268.4 ms (3 runs).** Row-major streaming is noisy, but col-major is never better. |
| **Reading** | The guidance matches the vendor's benchmark, which measures copy mode only ([EX] uses `streaming=False` throughout). It doesn't carry over to streaming into a per-PE state machine. |

## 2. Which axis is fastest in column-major

| | |
|---|---|
| **Vendor prose** | [SDK] Row-major: "l is the fastest varying dimension". Column-major: "w is the fastest varying dimension". |
| **Vendor code** | [EX] `hwl_2_oned_colmajor` (`run.appliance.py:178`) loops `for l / for w / for h`, so **h** is fastest. |
| **Measured** | The code is right and the prose is wrong. Layouts built per the code validate on hardware: 16 PEs at 5.32e-08, 65,536 PEs at 1.49e-07. Unit tests in `a775c10` fail on the prose's layout. |

## 3. Number of I/O channels

| | |
|---|---|
| **Vendor** | [SDK] "On WSE-3, there are a total of 62 channels at each edge." / "SdkRuntime supports up to 16 I/O channels". [CLI] `--channels`: "If this argument is not present, or is 0, then the previous single-I/O channel version is compiled." No performance guidance. |
| **Unsourced claim** | Our commits `24d995a` and `4040625` quote a "documented" statement that "performance improvements are minimal past 8 channels". **No such sentence exists** in [SDK] or [CLI] as of 2026-10-07. Do not cite it. |
| **Measured, copy mode** | Scales: 0.47 (1 ch) → 3.65–4.15 (8 ch) → 7.12–7.23 GB/s (16 ch), row-major. Still rising at the SDK's limit of 16 of the 62 physical channels per edge. **Verified (row-major): 0.33 GB/s (1 ch), 2.76 (8 ch), 5.20 (16 ch).** |
| **Measured, streaming** | Host-return times looked flat from 8 to 16 channels (20.3 → 20.0 ms). That supported "consumption-bound", but those are lower bounds. **Verified, header, full, row-major: 535.2 ms (1 ch), 152.1 ms (8 ch), 85.9 ms (16 ch, best of 3). Streaming scales 6.2× from 1 to 16 channels; the 'flat' reading was a measurement artifact.** In column mode: 106.9 ms (1 ch), 56.9 ms (8 ch, then a hang on the repeat), 65.6/66.5 ms (16 ch, col-major setup). |

## 4. Blocking semantics

| | |
|---|---|
| **Vendor** | [SDK] "If blocking mode (nonblock=False) is selected, the API waits until the operation is done. Otherwise, the function returns before the operation even starts." |
| **Measured** | `launch("wait_ew", nonblock=False)` returned before the device had finished. The next 0.46 MB D2H then took 27–58 ms at 8–16 channels and 193–371 ms at 1 channel, against 1.0 ms on an idle device. "Done" here means the launched function has returned, not that the command stream it blocked has been released. Our `wait_ew` returns early by design and holds the stream via `unblock_cmd_stream`. A blocking copy-mode `memcpy_h2d` also returns early: at 65,536 PEs the setup barrier adds 11–15 ms at 8 and 16 channels (e.g. 46.3 → 58.0 ms) and 75 ms at 1 channel. The barrier's own idle cost is 0.2 ms. |
| **Consequence** | Host-side timing of any kernel that releases the command stream asynchronously needs an explicit completion barrier. On-device timing (tsc) is the robust alternative, and it's what the vendor's own benchmarks use. |

## 5. System I/O capacity

| | |
|---|---|
| **Vendor** | [SYS] "1.2 Tb/s system I/O", provided by "12x100 GbE (4x QSFP-DD 4x100G)", i.e. 150 GB/s aggregate and 12.5 GB/s per link. Not yet pinned against a CS-3 datasheet PDF. |
| **Measured** | Fastest verified H2D: **6.75 GB/s** (copy, col-major setup, 16 ch). That's 4.5% of the 150 GB/s aggregate, or 54% of one 100 GbE link. Fastest verified streaming: 1.97 GB/s (column mode, 8 ch). The 10.35 GB/s from 6 Oct is a host-return figure and overstates the rate. Always quote it with the denominator. The draft's "1.3–1.5 GB/s flat" figures were measured with column-mode transfers at 1 channel, and they are host-return lower bounds on time too, so the true rates are lower still. |

## 6. Simulator vs hardware

| | |
|---|---|
| **Vendor** | The SDK presents the simulator (`SdkLauncher(simulator=True)`) as the development path for the same compiled program. |
| **Measured** | The behaviour diverges under concurrent streams. The bulk-`@fmovs` val kernel **passes in the simulator** in column mode (4×4, 5.32e-08) but **hangs on hardware** (DURATION 112 s and 103 s against ~20 s normal). It also hangs at 65K PEs with 8 channels. Whatever causes the hang, the simulator doesn't model it. |
| **Also** | One header column-mode run (8 channels) hung for 5 min 22 s on one system and was cancelled; the same config passed in 56.9 ms on another. Afterwards the first system failed even trivial 16-PE jobs (hangs and "Failed to produce fabric information from system") while other systems ran normally. So the hang is attributed to a faulty system, not to column mode, until it repeats on a healthy one. |
| **Also** | `SdkLauncher(simulator=True)` refuses a command that contains `%CMADDR%`; fixed in `08dee4f`. |

## 7. Attribution of the "interface, not hardware" reading

| | |
|---|---|
| **Vendor** | [TICKET] A support reply suggested that the HPC SDK transfer path wasn't optimised the way the tensor path was, and that lower-level interfaces exist below what is exposed. |
| **Status** | Not citable as is. The draft's `#todo` at `main.typ:208` stands: cite it, paraphrase it with permission, or drop it. Our measurements support only "what this interface, used this way, delivers". |
