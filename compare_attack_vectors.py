#!/usr/bin/env python3
"""
Script to compare different attack vectors across OSI layers

This script runs multiple simulations with different configurations to compare
the effectiveness of various attack vectors across different OSI layers.
"""

import argparse
import json
import time
import logging
import sys
from datetime import datetime
from tabulate import tabulate
from typing import List, Dict, Any

from network_simulation_framework import (
    NetworkSimulator, SimulationConfig, NetworkTarget,
    SimulationMode, TrafficPattern, OSILayer
)

# Define test scenarios
SCENARIOS = [
    {
        "name": "Network Layer Focus",
        "description": "Concentrates traffic on OSI Layer 3 (Network)",
        "layer_distribution": {
            OSILayer.NETWORK: 0.7,
            OSILayer.TRANSPORT: 0.1,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.05,
            OSILayer.APPLICATION: 0.05
        },
        "traffic_pattern": TrafficPattern.BURST,
        "intensity": 0.7
    },
    {
        "name": "Transport Layer Focus",
        "description": "Concentrates traffic on OSI Layer 4 (Transport)",
        "layer_distribution": {
            OSILayer.NETWORK: 0.1,
            OSILayer.TRANSPORT: 0.7,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.05,
            OSILayer.APPLICATION: 0.05
        },
        "traffic_pattern": TrafficPattern.RANDOM,
        "intensity": 0.7
    },
    {
        "name": "Session Layer Focus",
        "description": "Concentrates traffic on OSI Layer 5 (Session)",
        "layer_distribution": {
            OSILayer.NETWORK: 0.05,
            OSILayer.TRANSPORT: 0.15,
            OSILayer.SESSION: 0.7,
            OSILayer.PRESENTATION: 0.05,
            OSILayer.APPLICATION: 0.05
        },
        "traffic_pattern": TrafficPattern.UNIFORM,
        "intensity": 0.7
    },
    {
        "name": "Presentation Layer Focus",
        "description": "Concentrates traffic on OSI Layer 6 (Presentation)",
        "layer_distribution": {
            OSILayer.NETWORK: 0.05,
            OSILayer.TRANSPORT: 0.1,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.7,
            OSILayer.APPLICATION: 0.05
        },
        "traffic_pattern": TrafficPattern.GRADUAL,
        "intensity": 0.7
    },
    {
        "name": "Application Layer Focus",
        "description": "Concentrates traffic on OSI Layer 7 (Application)",
        "layer_distribution": {
            OSILayer.NETWORK: 0.05,
            OSILayer.TRANSPORT: 0.05,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.1,
            OSILayer.APPLICATION: 0.7
        },
        "traffic_pattern": TrafficPattern.ADAPTIVE,
        "intensity": 0.7
    },
    {
        "name": "Multi-layer Distributed",
        "description": "Distributes traffic evenly across all OSI layers",
        "layer_distribution": {
            OSILayer.NETWORK: 0.2,
            OSILayer.TRANSPORT: 0.2,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.2,
            OSILayer.APPLICATION: 0.2
        },
        "traffic_pattern": TrafficPattern.BURST,
        "intensity": 0.7
    },
    {
        "name": "Stealth Operations",
        "description": "Focuses on low-intensity, randomized traffic",
        "layer_distribution": {
            OSILayer.NETWORK: 0.3,
            OSILayer.TRANSPORT: 0.3,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.1,
            OSILayer.APPLICATION: 0.1
        },
        "traffic_pattern": TrafficPattern.RANDOM,
        "intensity": 0.3
    },
    {
        "name": "High Intensity Burst",
        "description": "High-intensity traffic with burst pattern",
        "layer_distribution": {
            OSILayer.NETWORK: 0.2,
            OSILayer.TRANSPORT: 0.2,
            OSILayer.SESSION: 0.2,
            OSILayer.PRESENTATION: 0.2,
            OSILayer.APPLICATION: 0.2
        },
        "traffic_pattern": TrafficPattern.BURST,
        "intensity": 0.9
    }
]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare different attack vectors across OSI layers"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Target URL or hostname"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=20,
        help="Duration of each test in seconds (default: 20)"
    )
    
    parser.add_argument(
        "--threads", "-n",
        type=int,
        default=10,
        help="Number of worker threads (default: 10)"
    )
    
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated list of scenario names to run (default: all)"
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

def run_scenario(scenario: Dict, target_host: str, duration: int, threads: int, verbose: bool) -> Dict:
    """Run a single test scenario"""
    print(f"\nRunning scenario: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    
    # Create target
    target = NetworkTarget(host=target_host)
    
    # Create configuration
    config = SimulationConfig()
    config.target = target
    config.mode = SimulationMode.ANALYZE
    config.duration = duration
    config.threads = threads
    config.layer_distribution = scenario["layer_distribution"]
    config.traffic_pattern = scenario["traffic_pattern"]
    config.intensity = scenario["intensity"]
    config.verbose = verbose
    
    # Disable adaptivity for consistent comparison
    config.adaptation_enabled = False
    
    # Create simulator
    simulator = NetworkSimulator(config)
    
    # Run simulation
    try:
        results = simulator.start()
        
        # Add scenario information
        results["scenario_name"] = scenario["name"]
        results["scenario_description"] = scenario["description"]
        results["target"] = target_host
        results["timestamp"] = datetime.now().isoformat()
        
        return results
    
    except KeyboardInterrupt:
        print("Interrupted by user")
        simulator.stop()
        sys.exit(1)

def main():
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Determine which scenarios to run
    scenario_names = []
    if args.scenarios:
        scenario_names = [name.strip() for name in args.scenarios.split(",")]
        
    scenarios_to_run = []
    if scenario_names:
        for scenario in SCENARIOS:
            if scenario["name"] in scenario_names:
                scenarios_to_run.append(scenario)
    else:
        scenarios_to_run = SCENARIOS
    
    if not scenarios_to_run:
        print(f"Error: No matching scenarios found. Available scenarios: {', '.join([s['name'] for s in SCENARIOS])}")
        return 1
    
    print(f"Target: {args.target}")
    print(f"Will run {len(scenarios_to_run)} scenarios, each for {args.duration} seconds")
    
    # Run the scenarios
    all_results = []
    for scenario in scenarios_to_run:
        result = run_scenario(
            scenario=scenario,
            target_host=args.target,
            duration=args.duration,
            threads=args.threads,
            verbose=args.verbose
        )
        all_results.append(result)
        
        # Wait briefly between tests
        time.sleep(2)
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
            print(f"\nDetailed results saved to {args.output}")
    
    # Analyze and display results
    table_data = []
    for result in all_results:
        table_data.append([
            result["scenario_name"],
            f"{result['operations_total']}",
            f"{result['success_rate'] * 100:.1f}%",
            f"{result['ops_per_second']:.2f}",
            f"{result['avg_response_time']:.2f}",
            f"{result['percentile_95']:.2f}",
        ])
    
    # Sort by ops/second (high to low)
    table_data.sort(key=lambda x: float(x[3]), reverse=True)
    
    # Print table
    print("\nResults Summary (sorted by throughput):")
    print(tabulate(
        table_data,
        headers=["Scenario", "Total Ops", "Success Rate", "Ops/sec", "Avg RT (ms)", "95th RT (ms)"],
        tablefmt="grid"
    ))
    
    # Determine best scenario for different metrics
    best_throughput = max(all_results, key=lambda x: x["ops_per_second"])
    best_success_rate = max(all_results, key=lambda x: x["success_rate"])
    best_response_time = min(all_results, key=lambda x: x["avg_response_time"])
    
    print("\nBest Scenarios:")
    print(f"- For Throughput: {best_throughput['scenario_name']} ({best_throughput['ops_per_second']:.2f} ops/sec)")
    print(f"- For Success Rate: {best_success_rate['scenario_name']} ({best_success_rate['success_rate'] * 100:.1f}%)")
    print(f"- For Response Time: {best_response_time['scenario_name']} ({best_response_time['avg_response_time']:.2f} ms)")
    
    # Calculate composite score (normalized)
    composite_scores = []
    
    # Normalize values
    max_ops = max(r["ops_per_second"] for r in all_results)
    max_success = max(r["success_rate"] for r in all_results)
    max_resp = max(r["avg_response_time"] for r in all_results)
    
    for result in all_results:
        # Normalize scores (0-1 where higher is better)
        ops_norm = result["ops_per_second"] / max_ops if max_ops > 0 else 0
        success_norm = result["success_rate"] / max_success if max_success > 0 else 0
        resp_norm = 1 - (result["avg_response_time"] / max_resp if max_resp > 0 else 0)
        
        # Calculate composite score (weighted)
        composite = (ops_norm * 0.4) + (success_norm * 0.4) + (resp_norm * 0.2)
        
        composite_scores.append({
            "scenario": result["scenario_name"],
            "score": composite
        })
    
    # Sort by composite score
    composite_scores.sort(key=lambda x: x["score"], reverse=True)
    
    print("\nOverall Effectiveness (composite score):")
    for i, score in enumerate(composite_scores[:3], 1):
        print(f"{i}. {score['scenario']} ({score['score']:.3f})")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)