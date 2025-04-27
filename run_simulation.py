#!/usr/bin/env python3
"""
Sample script to run a network simulation using the framework
"""

import argparse
import json
import logging
from datetime import datetime
from network_simulation_framework import (
    NetworkSimulator, SimulationConfig, NetworkTarget,
    SimulationMode, TrafficPattern, OSILayer
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run a network simulation test"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="https://httpbin.org/get",
        help="Target URL or hostname"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Test duration in seconds"
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["analyze", "benchmark", "stress", "stealth", "adaptive"],
        default="analyze",
        help="Simulation mode"
    )
    
    parser.add_argument(
        "--intensity", "-i",
        type=float,
        default=0.5,
        help="Traffic intensity (0.0-1.0)"
    )
    
    parser.add_argument(
        "--threads", "-n",
        type=int,
        default=10,
        help="Number of worker threads"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create the target
    target = NetworkTarget(
        host=args.target
    )
    
    # Create the configuration
    config = SimulationConfig()
    config.target = target
    config.mode = SimulationMode[args.mode.upper()]
    config.duration = args.duration
    config.threads = args.threads
    config.intensity = args.intensity
    config.verbose = args.verbose
    
    # Set up layer distribution based on mode
    if config.mode == SimulationMode.ANALYZE:
        # Balanced approach for analysis
        config.layer_distribution = {
            OSILayer.NETWORK: 0.2,
            OSILayer.TRANSPORT: 0.2,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.2,
            OSILayer.APPLICATION: 0.2
        }
    elif config.mode == SimulationMode.BENCHMARK:
        # Focus on application layer for benchmarking
        config.layer_distribution = {
            OSILayer.NETWORK: 0.1,
            OSILayer.TRANSPORT: 0.1,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.1,
            OSILayer.APPLICATION: 0.6
        }
    elif config.mode == SimulationMode.STRESS:
        # Distribute evenly for stress testing
        config.layer_distribution = {
            OSILayer.NETWORK: 0.25,
            OSILayer.TRANSPORT: 0.25,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.1,
            OSILayer.APPLICATION: 0.2
        }
    elif config.mode == SimulationMode.STEALTH:
        # Focus on lower layers for stealth
        config.layer_distribution = {
            OSILayer.NETWORK: 0.3,
            OSILayer.TRANSPORT: 0.3,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.1,
            OSILayer.APPLICATION: 0.1
        }
    elif config.mode == SimulationMode.ADAPTIVE:
        # Start balanced, will adapt based on responses
        config.layer_distribution = {
            OSILayer.NETWORK: 0.2,
            OSILayer.TRANSPORT: 0.2,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.2,
            OSILayer.APPLICATION: 0.2
        }
    
    # Set traffic pattern based on mode
    if config.mode == SimulationMode.ADAPTIVE:
        config.traffic_pattern = TrafficPattern.ADAPTIVE
    elif config.mode == SimulationMode.STEALTH:
        config.traffic_pattern = TrafficPattern.RANDOM
    elif config.mode == SimulationMode.STRESS:
        config.traffic_pattern = TrafficPattern.BURST
    elif config.mode == SimulationMode.BENCHMARK:
        config.traffic_pattern = TrafficPattern.UNIFORM
    else:
        config.traffic_pattern = TrafficPattern.GRADUAL
    
    # Validate the configuration
    valid, message = config.validate()
    if not valid:
        logging.error(f"Invalid configuration: {message}")
        return 1
    
    # Create the simulator
    simulator = NetworkSimulator(config)
    
    # Run the simulation
    print(f"Starting {args.mode} simulation against {args.target}")
    print(f"Duration: {args.duration}s, Threads: {args.threads}, Intensity: {args.intensity}")
    
    try:
        results = simulator.start()
        
        # Add additional information to results
        results["target"] = str(target)
        results["mode"] = config.mode.name
        results["timestamp"] = datetime.now().isoformat()
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"Results saved to {args.output}")
        
        # Print final summary
        print("\nSummary:")
        print(f"Operations: {results['operations_total']} total, {results['operations_success']} successful")
        print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
        print(f"Throughput: {results['ops_per_second']:.2f} ops/sec")
        print(f"Response Time: {results['avg_response_time']:.2f} ms (avg), {results['percentile_95']:.2f} ms (95th)")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        simulator.stop()
    
    return 0

if __name__ == "__main__":
    exit(main())