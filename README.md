# Network Simulation Framework

A terminal-based Python simulation framework to evaluate network behavior under diverse protocol loads across OSI layers 3 through 7, with adaptive traffic modeling, encrypted payload construction, and autonomous strategy switching — intended for stress-testing infrastructure performance and resilience in both internal and external environments.

## Features

- **Multi-layer Testing**: Evaluate traffic patterns across all OSI layers (3-7)
- **Terminal UI**: Real-time visualization of test progress and metrics
- **Adaptive Traffic Modeling**: Automatically adjust testing parameters based on target response
- **Multiple Traffic Patterns**: Uniform, burst, gradual, random, and adaptive patterns
- **Detailed Metrics**: Response times, success rates, throughput, etc.
- **Comparative Analysis**: Compare different attack vectors and traffic patterns
- **Protocol Support**: HTTP, HTTPS, DNS, TCP, UDP, ICMP, raw IP
- **Stealth Operations**: Techniques to minimize detection during testing

## Requirements

- Python 3.8+
- Required packages: 
  - curses (built-in on most Unix systems)
  - socket, ssl, struct (Python standard library)
  - tabulate (for results reporting)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install tabulate
```

## Usage

### Basic Usage

Run a simple simulation:

```bash
python run_simulation.py --target https://example.com --duration 30 --mode analyze
```

### Command Line Arguments

- `--target`, `-t`: Target URL or hostname
- `--duration`, `-d`: Test duration in seconds (default: 30)
- `--mode`, `-m`: Simulation mode (analyze, benchmark, stress, stealth, adaptive)
- `--intensity`, `-i`: Traffic intensity from 0.0 to 1.0 (default: 0.5)
- `--threads`, `-n`: Number of worker threads (default: 10)
- `--output`, `-o`: Output file for results (JSON)
- `--verbose`, `-v`: Enable verbose output

### Simulation Modes

- **analyze**: Balanced testing across all layers
- **benchmark**: Focus on measuring performance metrics
- **stress**: Maximum load testing
- **stealth**: Low-visibility testing with randomized patterns
- **adaptive**: Self-adjusting behavior based on target response

### Compare Attack Vectors

To compare different attack vectors and traffic patterns:

```bash
python compare_attack_vectors.py --target https://example.com --duration 20
```

This will run a series of tests with different configurations and provide a comparative analysis.

## Custom Test Scenarios

You can define custom test scenarios by modifying the `SCENARIOS` list in `compare_attack_vectors.py`.

## Framework Architecture

The framework consists of several key components:

1. **NetworkSimulator**: Main simulation controller
2. **TrafficGenerator**: Generates timing schedules and operation parameters
3. **NetworkOperator**: Executes network operations
4. **AdaptiveController**: Implements adaptive behavior
5. **TerminalUI**: Handles terminal visualization
6. **SimulationStats**: Collects and analyzes statistics

## OSI Layer Operations

The framework supports operations at each OSI layer:

- **Layer 3 (Network)**: ICMP echo requests, raw IP packets
- **Layer 4 (Transport)**: TCP/UDP socket operations
- **Layer 5 (Session)**: Session establishment, continuation, and termination
- **Layer 6 (Presentation)**: TLS handshakes, encoding transformations
- **Layer 7 (Application)**: HTTP requests, DNS queries

## Note on Privileges

Some network operations (particularly at OSI layers 3 and 4) may require elevated privileges to function properly. To use raw sockets for ICMP or IP packet operations, the script should be run with appropriate permissions.

## Legal and Ethical Use

This framework is intended for legitimate security testing, performance evaluation, and research purposes only. Always ensure you have proper authorization before testing any systems or networks you don't own.

## License

This project is released as open source software.