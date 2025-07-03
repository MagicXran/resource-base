# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreeFEM hot rolling simulation project for thermal-mechanical coupling analysis in metallurgical processes. This repository focuses on finite element method (FEM) simulation of steel rolling operations with temperature-dependent material properties and contact mechanics.

## FreeFEM Development Commands

### Installation Verification
```bash
# Check FreeFEM installation
FreeFem++ -h

# Verify version
FreeFem++ -v

# Test simple script execution
FreeFem++ script.edp -v 0 -nw
```

### Running Simulations
```bash
# Execute FreeFEM script with minimal output
FreeFem++ rolling_simulation.edp -v 0 -nw

# Run with specific parameters file
FreeFem++ thermal_mechanical.edp -f parameters.dat

# Parallel execution (if MPI configured)
mpirun -np 4 FreeFem++-mpi parallel_rolling.edp
```

### Python Integration Commands
```bash
# Install required Python dependencies
pip install fastapi uvicorn pydantic scipy numpy psutil redis

# Run FastAPI server for simulation API
uvicorn simulation_api:app --host 0.0.0.0 --port 8000 --reload

# Execute parametric studies
python optimization_runner.py --config rolling_params.json

# Generate simulation reports
python generate_report.py --results output/ --format pdf
```

## Core Architecture

### FreeFEM Simulation Structure

**Thermal-Mechanical Coupling**: The core simulation implements coupled physics through separate finite element spaces for temperature (P1) and displacement (P2) fields. Temperature affects material properties and thermal stress, while mechanical deformation generates heat through plastic work and friction.

**Rolling Geometry**: Contact mechanics between rolls and workpiece uses penalty methods or Lagrange multipliers for contact constraints. The geometry includes cylindrical rolls, deformable strip, and entry/exit zones with appropriate boundary conditions.

**Material Models**: Temperature-dependent properties include:
- Young's modulus: E(T) for thermal stress coupling
- Thermal conductivity: k(T) for heat transfer
- Plastic flow stress: σ_y(T,ε̇) for deformation resistance
- Thermal expansion: α(T) for thermal strain

### Python API Integration

**Subprocess Management**: FreeFEM execution through Python subprocess module with timeout handling and process monitoring. Windows-specific process group creation prevents orphaned processes.

**Parameter Exchange**: Template-based parameter substitution allows dynamic generation of FreeFEM scripts from Python configuration objects. Results parsing extracts key metrics like rolling force, temperature distribution, and stress fields.

**Background Processing**: FastAPI framework provides asynchronous job queuing for long-running simulations with real-time status monitoring and results retrieval.

### Key File Patterns

- `*.edp` - FreeFEM script files containing finite element formulations
- `*_template.edp` - Template scripts with parameter placeholders (${parameter})
- `parameters_*.json` - Configuration files for simulation parameters
- `results_*.dat` - Output data files from FreeFEM simulations
- `*_api.py` - FastAPI endpoints for simulation services
- `*_config.py` - Parameter validation and configuration classes

## Rolling Simulation Parameters

### Critical Process Parameters
- **Roll radius**: 0.3-0.8m (affects contact geometry and force)
- **Reduction ratio**: 10-80% (thickness reduction per pass)
- **Rolling speed**: 1-20 m/s (affects strain rate and temperature)
- **Initial temperature**: 1000-1300K (hot rolling temperature range)
- **Material grade**: Carbon steel, stainless steel, aluminum alloys

### Mesh Generation Strategy
- **Contact zone refinement**: High element density at roll-strip interface
- **Adaptive refinement**: Stress-based indicators for mesh adaptation
- **Element types**: P1 for temperature, P2 for displacement
- **Boundary layer**: Fine mesh near surfaces for heat transfer

### Validation Benchmarks
Industrial validation shows 3-10% accuracy for rolling force prediction and ±15°C for temperature when compared to experimental data from plate mills and laboratory rolling stands.

## Development Guidelines

### FreeFEM Script Development
1. Use separate function spaces for thermal and mechanical problems
2. Implement contact mechanics through penalty methods or Lagrange multipliers
3. Include material property temperature dependence in constitutive equations
4. Apply appropriate boundary conditions for rolling geometry

### Python Integration Patterns
1. Validate all simulation parameters using Pydantic models
2. Implement comprehensive error handling for subprocess execution
3. Use template substitution for dynamic parameter injection
4. Cache simulation results for optimization workflows

### Performance Optimization
1. Use adaptive mesh refinement for large deformation problems
2. Implement parallel processing for multi-pass rolling simulations
3. Apply result caching for parametric studies
4. Monitor system resources during long-running simulations

## File Organization

```
/
├── FreeFEM方案.md              # Comprehensive implementation guide
├── FreeFem++-4.15-win64.exe    # FreeFEM installer for Windows
├── FreeFEM-documentation.pdf   # Official FreeFEM documentation
└── CLAUDE.md                   # This development guide
```

This repository serves as a reference implementation and documentation source for FreeFEM-based hot rolling simulations with Python API integration.