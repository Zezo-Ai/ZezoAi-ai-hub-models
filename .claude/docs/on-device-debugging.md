# On-Device Debugging Guide

Reference this document when compilation or profiling fails on device.

## Downloading Debug Artifacts

Use `job.download_job_logs(output_dir)` to download runtime logs and other debug info for a failed job. This downloads both device logs and hub logs to the specified directory.

For full API documentation, see the AI Hub Workbench docs at https://workbench.aihub.qualcomm.com/docs/

## Common Failure Patterns

### Rank errors — `QnnDsp <E> [4294967295] has incorrect Rank 6`
- The Hexagon DSP only supports tensors up to rank 5
- Common sources: any reshape/view/permute that creates a 6D+ intermediate tensor
- Fix by monkeypatching the offending functions to restructure reshapes/permutations to stay at rank ≤ 5
- Verify patches produce numerically identical output before deploying

### Memory exceeded — `Failed to profile the model because memory usage exceeded device limits`
- Check the log for `unable to tile ... sufficiently` warnings — these show which ops have oversized intermediates
- Check for `std::bad_alloc` or `graph prepare failed` — total graph memory exhausted
- Common causes:
  - Resolution too high for the model's memory profile
  - Graph too large (traced loops duplicate ops and weights per iteration)
  - Large intermediate tensors that the DSP tiler cannot split

### DSP crash — `dspservice just died`
- Usually caused by memory pressure from the above issues
- The DSP silently dies rather than returning an error

## Resolution Search

Always try the model's native/default resolution first. Only if profiling fails (memory exceeded, DSP crash, timeout), then search for the maximum working resolution:

1. Write a script that traces the model (or each component, if it's a collection model), submits compile+profile jobs at multiple resolutions in parallel
2. Target device: Samsung Galaxy S25 (Family) for Snapdragon 8 Elite
3. Test a range of resolutions in a single batch to save time
4. Memory scaling varies by architecture — some models scale linearly with resolution, others quadratically or worse depending on attention/correlation mechanisms
5. Set the default resolution to the highest that profiles successfully
6. Consider whether a smaller model variant at higher resolution might be better than a larger variant at lower resolution

## QNN SDK Reference

Runtime logs and error codes can be interpreted using the QNN SDK documentation. If `$QNN_SDK_ROOT` is not set, ask the user for the path to their QNN SDK installation. Search within the provided directory for a folder matching `qnn_sdk-*` and set `QNN_SDK_ROOT` to that path. The docs are at `$QNN_SDK_ROOT/auto/docs/QNN/`.

Key docs:
- `general/htp/htp_backend.html` — HTP backend configuration, VTCM limits
- `HTP/scheduling_and_allocation.html` — how ops are scheduled and memory allocated
- `TfLite-Delegate/` — TFLite delegate behavior and limitations
