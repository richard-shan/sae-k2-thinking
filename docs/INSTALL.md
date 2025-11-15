# Installation & Setup Notes

## After Cloning

Make sure to make the shell scripts executable:

```bash
chmod +x setup.sh launch_pilot.sh launch_collection.sh monitor.sh
```

Then run setup:

```bash
./setup.sh
```

## Quick Test

Test that everything is working:

```bash
# Test Python imports
python3 -c "import torch; import transformers; print('✓ Imports OK')"

# Check GPU
nvidia-smi

# Test data loading (after collection)
python3 -c "from utils.dataset import ActivationDataset; print('✓ Utils OK')"
```

## File Permissions

All `.sh` files should be executable (`-rwxr-xr-x`).

If you get "Permission denied" when running bash scripts:
```bash
chmod +x *.sh
```

## Directory Structure

After setup, you should have:
```
kimi-sae-project/
├── data/           (empty, will be populated during collection)
├── logs/           (empty, will contain logs)
├── models/         (empty, will contain trained SAEs)
├── scripts/        (Python scripts - ready to use)
├── utils/          (Python modules - ready to import)
├── examples/       (Example scripts)
└── *.sh            (Bash scripts - should be executable)
```

## First Steps

1. **Setup**: `bash setup.sh`
2. **Pilot**: `bash launch_pilot.sh`  
3. **Monitor**: `bash monitor.sh` (in separate terminal during collection)
4. **Collect**: `bash launch_collection.sh`
5. **Train**: `python scripts/train_sae.py ...`

## Documentation

- **QUICKSTART.md** - Copy-paste commands for fast start
- **README.md** - Main documentation and overview
- **USAGE.md** - Detailed usage guide with examples
- **PROJECT_SUMMARY.md** - Complete technical reference

## Support

If you encounter issues:
1. Check the logs: `tail -f logs/shard0.log`
2. Verify setup: `bash setup.sh` again
3. Check USAGE.md troubleshooting section
4. Open a GitHub issue

---

Ready to start? Run: `bash setup.sh`
