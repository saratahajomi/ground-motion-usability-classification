# Included waveform examples

## Install without changing your existing interpreter

Open a terminal in the repository root. On Windows, using an installed Python 3.13:

```powershell
py -3.13 -m venv .venv-gm
.\.venv-gm\Scripts\python.exe -m pip install ./python_library
```

On macOS/Linux:

```bash
python3 -m venv .venv-gm
.venv-gm/bin/python -m pip install ./python_library
```

## Run both modes on the TXT input

```powershell
.\.venv-gm\Scripts\python.exe examples/test_usability_library.py examples/20031222191507_NC_PMM_HNN.txt
```

## Run both modes on the MiniSEED input

```powershell
.\.venv-gm\Scripts\python.exe examples/test_usability_library.py examples/60_antarctica2012_10deg.miniseed
```

The Antarctica input has 53 stored traces. Two contain 40 samples, below the required 64, and are skipped with warnings. The remaining 51 traces are processed separately. There is no merging or resampling.

The script saves four CSVs per input in `library_results/`: full/reduced predictions and full/reduced features. Use `--output-dir PATH` to choose another folder. Labels are 1 = usable and 0 = non-usable. The decision threshold is 0.91.

Full mode generates 1,368 features. Reduced mode generates only the embedded top 500 and uses the saved imputer for the remaining 868 model inputs. Reduced probabilities and labels can differ from full mode. Neither mode retrains anything.

## PyCharm

Keep your existing project interpreter unchanged. Use PyCharm's Terminal to run the commands above with `.venv-gm`'s Python. Alternatively, edit `INPUT_FILE` in `test_usability_library.py` and run that script using the separate environment. Without a command-line input, the script defaults to the included TXT file.

`input_manifest.json` records the input file sizes and SHA-256 hashes. Both inputs are exact copies of the files supplied for this release. They are example waveforms, not the complete training/testing data release. Third-party waveform data remain subject to their original providers' terms.

`library_outputs/` contains previously generated sample prediction outputs; filenames identify their source records.

## API documentation

See [the detailed function guide](../python_library/FUNCTION_GUIDE.md) for full and reduced behavior, imputation, and every library function.
