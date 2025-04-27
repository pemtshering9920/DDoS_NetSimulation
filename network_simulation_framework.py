#!/usr/bin/env python3
"""
Network Simulation Framework

A terminal-based Python simulation framework to evaluate network behavior under diverse
protocol loads across OSI layers 3 through 7, with adaptive traffic modeling, encrypted
payload construction, and autonomous strategy switching - intended for stress-testing
infrastructure performance and resilience in both internal and external environments.

This framework provides comprehensive network interaction capabilities and visualizes
traffic patterns, response metrics, and adaptation strategies directly in the terminal.
"""

import argparse
import random
import time
import threading
import socket
import ssl
import json
import hashlib
import struct
import ipaddress
import logging
import sys
import os
import curses
import math
import queue
import signal
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from urllib.parse import urlparse

# Terminal UI
import curses
from curses import wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='network_simulation.log'
)
logger = logging.getLogger("NetworkSimulator")

# Suppress verbose logging from libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)

class OSILayer(Enum):
    """OSI Layers supported by the simulation framework"""
    NETWORK = 3
    TRANSPORT = 4
    SESSION = 5
    PRESENTATION = 6
    APPLICATION = 7

class TrafficPattern(Enum):
    """Traffic patterns for network operations"""
    UNIFORM = auto()       # Constant rate
    BURST = auto()         # Bursts of activity
    GRADUAL = auto()       # Gradually increasing/decreasing
    RANDOM = auto()        # Random timing
    ADAPTIVE = auto()      # Self-adjusting based on responses

class OperationStatus(Enum):
    """Status of an operation"""
    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()
    BLOCKED = auto()

class SimulationMode(Enum):
    """Simulation modes"""
    ANALYZE = auto()       # Analyze target behavior
    BENCHMARK = auto()     # Benchmark performance
    STRESS = auto()        # Stress test
    STEALTH = auto()       # Stealth operations
    ADAPTIVE = auto()      # Adaptive behavior

class OperationResult:
    """Result of a network operation"""
    def __init__(self, 
                 status: OperationStatus,
                 response_time: float = 0.0,
                 status_code: Optional[int] = None,
                 response_size: Optional[int] = None,
                 error: Optional[str] = None):
        self.status = status
        self.response_time = response_time
        self.status_code = status_code
        self.response_size = response_size
        self.error = error
        self.timestamp = datetime.now()

class NetworkOperation:
    """Represents a network operation at a specific OSI layer"""
    def __init__(self, 
                 layer: OSILayer,
                 operation_id: str,
                 target_info: Dict[str, Any],
                 parameters: Dict[str, Any]):
        self.id = operation_id
        self.layer = layer
        self.target_info = target_info
        self.parameters = parameters
        self.result = None
        self.start_time = None
        self.end_time = None
        self.attempts = 0
        
    def execute(self) -> OperationResult:
        """Execute the operation"""
        self.start_time = datetime.now()
        self.attempts += 1
        
        # Placeholder for actual implementation
        logger.debug(f"Executing {self.layer.name} operation {self.id}")
        
        # Operation execution based on OSI layer
        result = None
        
        # Default result (for now)
        self.end_time = datetime.now()
        response_time = (self.end_time - self.start_time).total_seconds() * 1000  # ms
        self.result = OperationResult(
            status=OperationStatus.SUCCESS,
            response_time=response_time
        )
        return self.result

class NetworkTarget:
    """Target information for network operations"""
    def __init__(self, 
                 host: str,
                 port: int = None,
                 protocol: str = None,
                 path: str = "/",
                 requires_auth: bool = False,
                 ssl_enabled: bool = False):
        self.host = host
        self.path = path
        self.requires_auth = requires_auth
        self.ssl_enabled = ssl_enabled
        
        # Parse URL if provided as host
        if "://" in host:
            parsed = urlparse(host)
            self.host = parsed.netloc
            self.protocol = parsed.scheme
            self.path = parsed.path or "/"
            
            # Default ports based on protocol
            if port is None:
                if self.protocol == "https":
                    self.port = 443
                    self.ssl_enabled = True
                elif self.protocol == "http":
                    self.port = 80
                elif self.protocol == "dns":
                    self.port = 53
                else:
                    self.port = 80
        else:
            self.protocol = protocol or "http"
            self.port = port or (443 if self.ssl_enabled else 80)
            
        # Resolve IP address
        try:
            self.ip_address = socket.gethostbyname(self.host)
        except socket.gaierror:
            self.ip_address = "0.0.0.0"
            logger.warning(f"Could not resolve hostname: {self.host}")
        
        # Additional properties
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        self.status_codes = defaultdict(int)     # Count of status codes
        self.errors = defaultdict(int)           # Count of error types
        self.last_accessed = None
        self.total_requests = 0
        self.successful_requests = 0
        
    def update_stats(self, result: OperationResult):
        """Update target statistics based on operation result"""
        self.last_accessed = datetime.now()
        self.total_requests += 1
        
        if result.status == OperationStatus.SUCCESS:
            self.successful_requests += 1
            if result.response_time:
                self.response_times.append(result.response_time)
            if result.status_code:
                self.status_codes[result.status_code] += 1
        else:
            self.errors[result.status.name] += 1
            
    def get_success_rate(self) -> float:
        """Get the success rate of operations against this target"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_avg_response_time(self) -> float:
        """Get the average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def __str__(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

class SimulationConfig:
    """Configuration for a network simulation"""
    def __init__(self):
        # Simulation parameters
        self.mode = SimulationMode.ANALYZE
        self.duration = 60  # seconds
        self.target = None
        self.threads = 1
        self.timeout = 10.0
        self.verbose = False
        
        # Traffic parameters
        self.traffic_pattern = TrafficPattern.ADAPTIVE
        self.intensity = 0.5  # 0.0 to 1.0
        self.layer_distribution = {
            OSILayer.NETWORK: 0.1,
            OSILayer.TRANSPORT: 0.2,
            OSILayer.SESSION: 0.1,
            OSILayer.PRESENTATION: 0.2,
            OSILayer.APPLICATION: 0.4
        }
        
        # Adaptation parameters
        self.adaptation_enabled = True
        self.adaptation_threshold = 0.2
        self.learning_rate = 0.05
        
        # Authentication
        self.auth_username = None
        self.auth_password = None
        self.auth_token = None
        
        # SSL/TLS
        self.ssl_verify = False
        self.ssl_cert = None
        self.ssl_key = None
        
        # Advanced parameters
        self.randomization_factor = 0.3
        self.custom_headers = {}
        self.custom_payloads = {}
        self.max_retries = 3
        
    def from_args(self, args):
        """Initialize configuration from command line arguments"""
        if args.target:
            self.target = NetworkTarget(
                host=args.target,
                port=args.port,
                protocol=args.protocol,
                ssl_enabled=args.ssl
            )
        
        if args.mode:
            self.mode = SimulationMode[args.mode.upper()]
            
        if args.duration:
            self.duration = args.duration
            
        if args.threads:
            self.threads = args.threads
            
        if args.pattern:
            self.traffic_pattern = TrafficPattern[args.pattern.upper()]
            
        if args.intensity:
            self.intensity = float(args.intensity)
            
        # Additional parameters
        self.verbose = args.verbose
        self.timeout = args.timeout
        self.adaptation_enabled = not args.no_adapt
        self.auth_username = args.username
        self.auth_password = args.password
        self.ssl_verify = not args.no_verify
        
        return self
    
    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration"""
        if not self.target:
            return False, "No target specified"
        
        if self.intensity < 0.0 or self.intensity > 1.0:
            return False, "Intensity must be between 0.0 and 1.0"
        
        if self.duration <= 0:
            return False, "Duration must be positive"
        
        if self.threads <= 0:
            return False, "Thread count must be positive"
        
        if self.requires_auth and not (self.auth_username and self.auth_password) and not self.auth_token:
            return False, "Authentication required but credentials not provided"
            
        return True, "Configuration valid"
    
    @property
    def requires_auth(self) -> bool:
        """Check if the target requires authentication"""
        return self.target and self.target.requires_auth

class TrafficGenerator:
    """Generates timing schedules for network operations"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = random.Random()  # Use a dedicated random number generator
        
    def generate_timings(self, pattern: TrafficPattern, duration: float, intensity: float) -> List[float]:
        """
        Generate a list of timestamps for operations
        
        Args:
            pattern: The traffic pattern to use
            duration: Duration in seconds
            intensity: Value between 0.0 and 1.0
            
        Returns:
            List of timestamps (seconds from start)
        """
        intensity = max(0.1, min(1.0, intensity))
        timestamps = []
        
        if pattern == TrafficPattern.UNIFORM:
            # Evenly spaced operations
            # Base rate: 10 req/sec at max intensity
            rate = 10 * intensity
            interval = 1.0 / rate
            
            current = 0.0
            while current < duration:
                timestamps.append(current)
                current += interval
                
        elif pattern == TrafficPattern.BURST:
            # Bursts of activity followed by quiet periods
            current = 0.0
            while current < duration:
                # Burst parameters
                burst_duration = self.rng.uniform(1.0, 3.0)
                burst_rate = 20 * intensity  # Higher rate during bursts
                quiet_duration = self.rng.uniform(3.0, 8.0) / intensity
                
                # Add a burst of requests
                burst_end = min(current + burst_duration, duration)
                burst_interval = 1.0 / burst_rate
                
                burst_time = current
                while burst_time < burst_end:
                    timestamps.append(burst_time)
                    burst_time += burst_interval
                
                # Move to after the quiet period
                current = burst_end + quiet_duration
                
        elif pattern == TrafficPattern.GRADUAL:
            # Gradually increasing/decreasing intensity
            steps = int(duration / 5)  # Change intensity every 5 seconds
            
            for step in range(steps):
                step_start = step * 5
                step_end = min(step_start + 5, duration)
                step_duration = step_end - step_start
                
                # Intensity follows a sine wave pattern
                progress = step / steps
                current_intensity = intensity * (0.5 + 0.5 * math.sin(progress * math.pi))
                rate = max(1, 10 * current_intensity)
                interval = 1.0 / rate
                
                current = step_start
                while current < step_end:
                    timestamps.append(current)
                    current += interval
        
        elif pattern == TrafficPattern.RANDOM:
            # Random timing with average intensity
            rate = 10 * intensity  # Base rate at max intensity
            
            # Generate using Poisson process (exponential interarrival times)
            current = 0.0
            while current < duration:
                # Exponential distribution with mean 1/rate
                interval = self.rng.expovariate(rate)
                current += interval
                if current < duration:
                    timestamps.append(current)
                    
        elif pattern == TrafficPattern.ADAPTIVE:
            # Start with random pattern
            initial_timestamps = self.generate_timings(
                TrafficPattern.RANDOM, duration / 3, intensity
            )
            timestamps.extend(initial_timestamps)
            
            # Middle section with bursts
            mid_timestamps = self.generate_timings(
                TrafficPattern.BURST, duration / 3, intensity
            )
            # Offset timestamps
            mid_timestamps = [t + (duration / 3) for t in mid_timestamps]
            timestamps.extend(mid_timestamps)
            
            # Final section with gradual pattern
            final_timestamps = self.generate_timings(
                TrafficPattern.GRADUAL, duration / 3, intensity
            )
            # Offset timestamps
            final_timestamps = [t + (2 * duration / 3) for t in final_timestamps]
            timestamps.extend(final_timestamps)
            
        return sorted(timestamps)
    
    def select_layer(self) -> OSILayer:
        """Select an OSI layer for the next operation based on configuration"""
        distribution = self.config.layer_distribution
        layers = list(distribution.keys())
        weights = list(distribution.values())
        
        return self.rng.choices(layers, weights=weights, k=1)[0]
    
    def generate_operation_parameters(self, layer: OSILayer, target: NetworkTarget) -> Dict[str, Any]:
        """Generate parameters for an operation at the specified layer"""
        params = {"timestamp": datetime.now()}
        
        # Network layer (Layer 3)
        if layer == OSILayer.NETWORK:
            params.update({
                "protocol": self.rng.choice(["icmp", "ip"]),
                "ttl": self.rng.randint(32, 255),
                "packet_size": self.rng.randint(64, 1500),
                "flags": self.rng.choice(["df", "mf", None])
            })
            
            if params["protocol"] == "icmp":
                params.update({
                    "icmp_type": 8,  # Echo request
                    "icmp_code": 0
                })
                
        # Transport layer (Layer 4)
        elif layer == OSILayer.TRANSPORT:
            params.update({
                "protocol": self.rng.choice(["tcp", "udp"]),
                "source_port": self.rng.randint(10000, 65000),
                "dest_port": target.port
            })
            
            if params["protocol"] == "tcp":
                params.update({
                    "flags": self.rng.choice(["SYN", "ACK", "PSH", "RST", "FIN"]),
                    "window_size": self.rng.randint(1024, 65535)
                })
                
        # Session layer (Layer 5)
        elif layer == OSILayer.SESSION:
            params.update({
                "session_type": self.rng.choice(["new", "continue", "close"]),
                "keep_alive": self.rng.choice([True, False]),
                "session_id": f"session-{self.rng.randbytes(8).hex()}"
            })
            
        # Presentation layer (Layer 6)
        elif layer == OSILayer.PRESENTATION:
            params.update({
                "encoding": self.rng.choice(["json", "xml", "binary"]),
                "encryption": self.rng.choice(["tls", "none"]),
                "compression": self.rng.choice(["gzip", "deflate", "none"])
            })
            
            if params["encryption"] == "tls":
                params.update({
                    "tls_version": self.rng.choice(["1.2", "1.3"]),
                    "cipher": self.rng.choice([
                        "TLS_AES_256_GCM_SHA384", 
                        "TLS_CHACHA20_POLY1305_SHA256"
                    ])
                })
                
        # Application layer (Layer 7)
        elif layer == OSILayer.APPLICATION:
            if target.protocol in ["http", "https"]:
                params.update({
                    "method": self.rng.choice(["GET", "POST", "HEAD", "PUT"]),
                    "path": target.path,
                    "headers": {
                        "User-Agent": self._generate_random_user_agent(),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive"
                    }
                })
                
                # Add custom headers from config
                if self.config.custom_headers:
                    params["headers"].update(self.config.custom_headers)
                    
                # Add custom payloads for non-GET requests
                if params["method"] in ["POST", "PUT"] and self.config.custom_payloads:
                    payload_key = f"{target.host}_{params['method']}"
                    if payload_key in self.config.custom_payloads:
                        params["payload"] = self.config.custom_payloads[payload_key]
                    else:
                        params["payload"] = json.dumps({"test": "data", "timestamp": datetime.now().isoformat()})
                        
            elif target.protocol == "dns":
                params.update({
                    "query_type": self.rng.choice(["A", "AAAA", "MX", "TXT"]),
                    "domain": target.host,
                    "recursive": True
                })
                
        # Apply randomization based on config
        self._apply_randomization(params)
        
        return params
    
    def _generate_random_user_agent(self) -> str:
        """Generate a random user agent string"""
        templates = [
            "Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_ver} Safari/537.36",
            "Mozilla/5.0 ({os}) Gecko/20100101 Firefox/{firefox_ver}",
            "Mozilla/5.0 ({os}) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{safari_ver} Safari/605.1.15"
        ]
        
        os_strings = [
            "Windows NT 10.0; Win64; x64",
            "Windows NT 6.1; Win64; x64",
            "Macintosh; Intel Mac OS X 10_15_7",
            "X11; Linux x86_64"
        ]
        
        template = self.rng.choice(templates)
        os_string = self.rng.choice(os_strings)
        
        chrome_ver = f"{self.rng.randint(90, 110)}.0.{self.rng.randint(1000, 5000)}.{self.rng.randint(10, 100)}"
        firefox_ver = f"{self.rng.randint(90, 110)}.0"
        safari_ver = f"{self.rng.randint(14, 17)}.{self.rng.randint(0, 9)}"
        
        return template.format(
            os=os_string,
            chrome_ver=chrome_ver,
            firefox_ver=firefox_ver,
            safari_ver=safari_ver
        )
    
    def _apply_randomization(self, params: Dict[str, Any]):
        """Apply randomization to parameters based on config"""
        if self.config.randomization_factor <= 0:
            return
            
        # Randomize user agent with certain probability
        if "headers" in params and self.rng.random() < self.config.randomization_factor:
            params["headers"]["User-Agent"] = self._generate_random_user_agent()
            
        # Randomize other parameters
        for key in list(params.keys()):
            # Only randomize numeric values with some probability
            if isinstance(params[key], (int, float)) and self.rng.random() < self.config.randomization_factor * 0.5:
                # Randomize by ±10%
                factor = 1.0 + (self.rng.random() * 0.2 - 0.1)
                params[key] = int(params[key] * factor) if isinstance(params[key], int) else params[key] * factor

class NetworkOperator:
    """Handles execution of network operations"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.target = config.target
        self.sockets = {}  # Persistent sockets by layer
        self.sessions = {}  # Persistent sessions
        self.ssl_contexts = {}  # SSL contexts by parameters
        
    def execute_operation(self, operation: NetworkOperation) -> OperationResult:
        """Execute a network operation"""
        logger.debug(f"Executing {operation.layer.name} operation")
        
        layer = operation.layer
        params = operation.parameters
        
        try:
            if layer == OSILayer.NETWORK:
                return self._execute_network_operation(params)
            elif layer == OSILayer.TRANSPORT:
                return self._execute_transport_operation(params)
            elif layer == OSILayer.SESSION:
                return self._execute_session_operation(params)
            elif layer == OSILayer.PRESENTATION:
                return self._execute_presentation_operation(params)
            elif layer == OSILayer.APPLICATION:
                return self._execute_application_operation(params)
            else:
                return OperationResult(
                    status=OperationStatus.FAILED,
                    error=f"Unsupported layer: {layer}"
                )
        except socket.timeout:
            return OperationResult(
                status=OperationStatus.TIMEOUT,
                error="Operation timed out"
            )
        except Exception as e:
            logger.error(f"Operation error: {str(e)}", exc_info=True)
            return OperationResult(
                status=OperationStatus.FAILED,
                error=str(e)
            )
    
    def _execute_network_operation(self, params: Dict[str, Any]) -> OperationResult:
        """Execute a network layer operation"""
        protocol = params.get("protocol", "icmp")
        target_ip = self.target.ip_address
        
        start_time = time.time()
        
        try:
            if protocol == "icmp":
                # Create socket for ICMP
                icmp_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
                icmp_socket.settimeout(self.config.timeout)
                
                # Create ICMP packet
                icmp_type = params.get("icmp_type", 8)
                icmp_code = params.get("icmp_code", 0)
                icmp_checksum = 0
                icmp_id = params.get("identifier", os.getpid() & 0xFFFF)
                icmp_seq = params.get("sequence", 1)
                
                # Create header and calculate checksum
                header = struct.pack("!BBHHH", icmp_type, icmp_code, icmp_checksum, icmp_id, icmp_seq)
                
                # Add payload to reach desired size
                payload_size = max(0, params.get("packet_size", 64) - 8)  # 8 bytes for ICMP header
                payload = bytes([i & 0xff for i in range(payload_size)])
                
                # Calculate checksum
                packet = header + payload
                checksum = self._calculate_checksum(packet)
                
                # Create final packet with correct checksum
                header = struct.pack("!BBHHH", icmp_type, icmp_code, checksum, icmp_id, icmp_seq)
                packet = header + payload
                
                # Send packet
                icmp_socket.sendto(packet, (target_ip, 0))
                
                # Wait for response (if echo request)
                if icmp_type == 8:  # Echo request
                    recv_packet, addr = icmp_socket.recvfrom(1024)
                    icmp_header = recv_packet[20:28]  # Extract ICMP header from IP packet
                    recv_type, recv_code, recv_checksum, recv_id, recv_seq = struct.unpack("!BBHHH", icmp_header)
                    
                    # Check if response matches request
                    if recv_type == 0 and recv_id == icmp_id and recv_seq == icmp_seq:
                        status = OperationStatus.SUCCESS
                    else:
                        status = OperationStatus.FAILED
                else:
                    # For non-echo requests, we can't easily verify success
                    status = OperationStatus.SUCCESS
                
                icmp_socket.close()
            else:
                # Simulate IP packet sending (actual raw IP would require root)
                # In a real implementation, this would use raw sockets
                logger.debug(f"Simulating IP packet to {target_ip}")
                time.sleep(0.01)  # Simulate network delay
                status = OperationStatus.SUCCESS
        
        except PermissionError:
            logger.warning("Raw socket operations require root privileges")
            status = OperationStatus.FAILED
            icmp_socket.close()
        except Exception as e:
            logger.error(f"Network operation error: {str(e)}")
            status = OperationStatus.FAILED
        
        end_time = time.time()
        
        return OperationResult(
            status=status,
            response_time=(end_time - start_time) * 1000  # ms
        )
    
    def _execute_transport_operation(self, params: Dict[str, Any]) -> OperationResult:
        """Execute a transport layer operation"""
        protocol = params.get("protocol", "tcp")
        target_ip = self.target.ip_address
        target_port = params.get("dest_port", self.target.port)
        source_port = params.get("source_port", 0)
        
        start_time = time.time()
        
        try:
            if protocol == "tcp":
                # Create TCP socket
                tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                tcp_socket.settimeout(self.config.timeout)
                if source_port > 0:
                    try:
                        tcp_socket.bind(('', source_port))
                    except:
                        pass  # Ignore binding errors, let the OS choose a port
                
                # Set TCP flags if specified
                flags = params.get("flags", "")
                if flags == "SYN":
                    # Just connect
                    result = tcp_socket.connect_ex((target_ip, target_port))
                    status = OperationStatus.SUCCESS if result == 0 else OperationStatus.FAILED
                elif flags == "FIN":
                    # Connect and then close properly
                    result = tcp_socket.connect_ex((target_ip, target_port))
                    if result == 0:
                        tcp_socket.shutdown(socket.SHUT_WR)
                        status = OperationStatus.SUCCESS
                    else:
                        status = OperationStatus.FAILED
                elif flags == "RST":
                    # Connect and then force RST
                    result = tcp_socket.connect_ex((target_ip, target_port))
                    if result == 0:
                        tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, 
                                             struct.pack('ii', 1, 0))
                        status = OperationStatus.SUCCESS
                    else:
                        status = OperationStatus.FAILED
                else:
                    # Standard connect
                    result = tcp_socket.connect_ex((target_ip, target_port))
                    status = OperationStatus.SUCCESS if result == 0 else OperationStatus.FAILED
                    
                tcp_socket.close()
            
            elif protocol == "udp":
                # Create UDP socket
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.settimeout(self.config.timeout)
                
                # Prepare data
                packet_size = params.get("packet_size", 64)
                data = b'X' * packet_size
                
                # Send data
                udp_socket.sendto(data, (target_ip, target_port))
                
                # Try to receive response (may not get one for UDP)
                try:
                    udp_socket.recvfrom(1024)
                    status = OperationStatus.SUCCESS
                except socket.timeout:
                    # For UDP, a timeout is often normal and doesn't indicate failure
                    status = OperationStatus.SUCCESS
                
                udp_socket.close()
        
        except Exception as e:
            logger.error(f"Transport operation error: {str(e)}")
            status = OperationStatus.FAILED
        
        end_time = time.time()
        
        return OperationResult(
            status=status,
            response_time=(end_time - start_time) * 1000  # ms
        )
    
    def _execute_session_operation(self, params: Dict[str, Any]) -> OperationResult:
        """Execute a session layer operation"""
        session_type = params.get("session_type", "new")
        session_id = params.get("session_id", "")
        
        start_time = time.time()
        
        # Session operations manage connection state
        try:
            if session_type == "new":
                # Create a new session
                if session_id in self.sessions:
                    # Session already exists, close it first
                    old_session = self.sessions[session_id]
                    if hasattr(old_session, 'close'):
                        old_session.close()
                
                # Create appropriate session based on protocol
                if self.target.protocol in ["http", "https"]:
                    # For HTTP, we'll use a persistent connection
                    session_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    session_socket.settimeout(self.config.timeout)
                    session_socket.connect((self.target.ip_address, self.target.port))
                    
                    if self.target.ssl_enabled:
                        ssl_context = ssl.create_default_context()
                        if not self.config.ssl_verify:
                            ssl_context.check_hostname = False
                            ssl_context.verify_mode = ssl.CERT_NONE
                        session_socket = ssl_context.wrap_socket(
                            session_socket, server_hostname=self.target.host
                        )
                    
                    # Store socket in sessions
                    self.sessions[session_id] = session_socket
                    status = OperationStatus.SUCCESS
                
                else:
                    # For other protocols, just simulate session creation
                    self.sessions[session_id] = {
                        "created": datetime.now(),
                        "last_activity": datetime.now(),
                        "protocol": self.target.protocol
                    }
                    status = OperationStatus.SUCCESS
                    
            elif session_type == "continue":
                # Continue existing session
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    
                    if isinstance(session, socket.socket):
                        # For socket-based sessions, send a keep-alive
                        if self.target.protocol in ["http", "https"]:
                            try:
                                # Send HTTP keep-alive request
                                request = (
                                    f"HEAD / HTTP/1.1\r\n"
                                    f"Host: {self.target.host}\r\n"
                                    f"Connection: keep-alive\r\n\r\n"
                                ).encode()
                                session.send(request)
                                
                                # Receive response
                                response = session.recv(1024)
                                if response:
                                    status = OperationStatus.SUCCESS
                                else:
                                    status = OperationStatus.FAILED
                            except:
                                # Session might be closed by server
                                status = OperationStatus.FAILED
                        else:
                            # For other protocols, just check if socket is still connected
                            try:
                                session.getpeername()
                                status = OperationStatus.SUCCESS
                            except:
                                status = OperationStatus.FAILED
                    else:
                        # For simulated sessions, just update timestamp
                        session["last_activity"] = datetime.now()
                        status = OperationStatus.SUCCESS
                else:
                    # Session doesn't exist
                    status = OperationStatus.FAILED
                    
            elif session_type == "close":
                # Close existing session
                if session_id in self.sessions:
                    session = self.sessions[session_id]
                    
                    if isinstance(session, socket.socket):
                        try:
                            session.close()
                        except:
                            pass
                    
                    # Remove from sessions
                    del self.sessions[session_id]
                    status = OperationStatus.SUCCESS
                else:
                    # Session doesn't exist
                    status = OperationStatus.SUCCESS  # Still success since end goal is achieved
            
            else:
                # Unknown session type
                status = OperationStatus.FAILED
        
        except Exception as e:
            logger.error(f"Session operation error: {str(e)}")
            status = OperationStatus.FAILED
        
        end_time = time.time()
        
        return OperationResult(
            status=status,
            response_time=(end_time - start_time) * 1000  # ms
        )
    
    def _execute_presentation_operation(self, params: Dict[str, Any]) -> OperationResult:
        """Execute a presentation layer operation"""
        encoding = params.get("encoding", "json")
        encryption = params.get("encryption", "none")
        
        start_time = time.time()
        
        try:
            if encryption == "tls":
                # TLS handshake operation
                tls_version = params.get("tls_version", "1.3")
                
                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.timeout)
                
                # Connect
                sock.connect((self.target.ip_address, self.target.port))
                
                # Create SSL context
                context = ssl.create_default_context()
                
                # Configure TLS version
                if tls_version == "1.2":
                    context.maximum_version = ssl.TLSVersion.TLSv1_2
                elif tls_version == "1.3":
                    context.minimum_version = ssl.TLSVersion.TLSv1_3
                
                # Configure verification
                if not self.config.ssl_verify:
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                
                # SSL handshake
                ssl_sock = context.wrap_socket(sock, server_hostname=self.target.host)
                
                # Get negotiated protocol version and cipher
                negotiated_version = ssl_sock.version()
                negotiated_cipher = ssl_sock.cipher()
                
                # Close connection
                ssl_sock.close()
                
                status = OperationStatus.SUCCESS
                
            else:
                # Simulate presentation operation for other encodings
                # This would typically handle data transformation, serialization, etc.
                time.sleep(0.01)  # Simulate processing
                status = OperationStatus.SUCCESS
        
        except Exception as e:
            logger.error(f"Presentation operation error: {str(e)}")
            status = OperationStatus.FAILED
        
        end_time = time.time()
        
        return OperationResult(
            status=status,
            response_time=(end_time - start_time) * 1000  # ms
        )
    
    def _execute_application_operation(self, params: Dict[str, Any]) -> OperationResult:
        """Execute an application layer operation"""
        start_time = time.time()
        status_code = None
        response_size = None
        
        try:
            if self.target.protocol in ["http", "https"]:
                method = params.get("method", "GET")
                path = params.get("path", "/")
                headers = params.get("headers", {})
                payload = params.get("payload", "")
                
                # Create socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.config.timeout)
                
                # Connect
                sock.connect((self.target.ip_address, self.target.port))
                
                # Apply SSL if needed
                if self.target.ssl_enabled:
                    context = ssl.create_default_context()
                    if not self.config.ssl_verify:
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(sock, server_hostname=self.target.host)
                
                # Build HTTP request
                request_lines = [f"{method} {path} HTTP/1.1", f"Host: {self.target.host}"]
                
                # Add headers
                for name, value in headers.items():
                    request_lines.append(f"{name}: {value}")
                
                # Add content-length for requests with payload
                if payload and method in ["POST", "PUT"]:
                    if isinstance(payload, str):
                        payload = payload.encode('utf-8')
                    request_lines.append(f"Content-Length: {len(payload)}")
                
                # Finish request
                request = "\r\n".join(request_lines) + "\r\n\r\n"
                if payload and method in ["POST", "PUT"]:
                    if isinstance(payload, str):
                        request = request.encode('utf-8') + payload.encode('utf-8')
                    else:
                        request = request.encode('utf-8') + payload
                else:
                    request = request.encode('utf-8')
                
                # Send request
                sock.sendall(request)
                
                # Receive response
                response_data = b""
                while True:
                    try:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        response_data += chunk
                    except socket.timeout:
                        break
                
                # Parse status code
                if response_data:
                    try:
                        status_line = response_data.split(b'\r\n')[0].decode('utf-8')
                        status_code = int(status_line.split(' ')[1])
                        response_size = len(response_data)
                        
                        # Determine operation status based on HTTP status code
                        if 200 <= status_code < 400:
                            status = OperationStatus.SUCCESS
                        elif 400 <= status_code < 500:
                            status = OperationStatus.BLOCKED
                        else:
                            status = OperationStatus.FAILED
                    except:
                        status = OperationStatus.FAILED
                else:
                    status = OperationStatus.FAILED
                
                # Close socket
                sock.close()
                
            elif self.target.protocol == "dns":
                # Simulate DNS query
                query_type = params.get("query_type", "A")
                domain = params.get("domain", self.target.host)
                
                # Use socket for real DNS query
                try:
                    if query_type == "A":
                        result = socket.gethostbyname(domain)
                        status = OperationStatus.SUCCESS
                    else:
                        # For other query types, simulate
                        time.sleep(0.05)
                        status = OperationStatus.SUCCESS
                except socket.gaierror:
                    status = OperationStatus.FAILED
            
            else:
                # Unknown protocol
                logger.warning(f"Unsupported protocol: {self.target.protocol}")
                status = OperationStatus.FAILED
        
        except Exception as e:
            logger.error(f"Application operation error: {str(e)}")
            status = OperationStatus.FAILED
        
        end_time = time.time()
        
        return OperationResult(
            status=status,
            response_time=(end_time - start_time) * 1000,  # ms
            status_code=status_code,
            response_size=response_size
        )
    
    def _calculate_checksum(self, data: bytes) -> int:
        """Calculate checksum for ICMP packet"""
        checksum = 0
        # Group data by 2 bytes (16 bits) and sum
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                checksum += (data[i] << 8) + data[i + 1]
            else:
                checksum += data[i] << 8
        
        # Add carry
        while checksum >> 16:
            checksum = (checksum & 0xFFFF) + (checksum >> 16)
        
        # Take one's complement
        return ~checksum & 0xFFFF
    
    def cleanup(self):
        """Clean up resources"""
        for session_id, session in list(self.sessions.items()):
            if isinstance(session, socket.socket):
                try:
                    session.close()
                except:
                    pass
        
        self.sessions.clear()

class SimulationStats:
    """Collects and analyzes simulation statistics"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.operations_total = 0
        self.operations_success = 0
        self.operations_failed = 0
        self.operations_timeout = 0
        self.operations_blocked = 0
        
        self.response_times = deque(maxlen=1000)
        self.status_codes = defaultdict(int)
        self.layer_stats = {layer: {"total": 0, "success": 0} for layer in OSILayer}
        
        self.adaptations = 0
        self.pattern_switches = 0
        
        # Time series data for graphing
        self.time_series = {
            "timestamps": [],
            "success_rate": [],
            "response_time": [],
            "operations_per_second": []
        }
        
        # Performance metrics
        self.peak_ops_per_second = 0
        self.avg_response_time = 0
        self.percentile_95 = 0
        self.success_rate = 0
        
    def start(self):
        """Start collecting statistics"""
        self.start_time = datetime.now()
    
    def stop(self):
        """Stop collecting statistics"""
        self.end_time = datetime.now()
        self._calculate_final_metrics()
    
    def record_operation(self, operation: NetworkOperation):
        """Record an operation result"""
        if not self.start_time:
            self.start_time = datetime.now()
            
        result = operation.result
        if not result:
            return
            
        self.operations_total += 1
        
        # Record by status
        if result.status == OperationStatus.SUCCESS:
            self.operations_success += 1
            self.layer_stats[operation.layer]["success"] += 1
        elif result.status == OperationStatus.TIMEOUT:
            self.operations_timeout += 1
        elif result.status == OperationStatus.BLOCKED:
            self.operations_blocked += 1
        else:  # FAILED
            self.operations_failed += 1
            
        self.layer_stats[operation.layer]["total"] += 1
        
        # Record response time
        if result.response_time:
            self.response_times.append(result.response_time)
            
        # Record status code
        if result.status_code:
            self.status_codes[result.status_code] += 1
            
        # Update time series every 10 operations
        if self.operations_total % 10 == 0:
            self._update_time_series()
    
    def record_adaptation(self):
        """Record an adaptation event"""
        self.adaptations += 1
    
    def record_pattern_switch(self):
        """Record a pattern switch event"""
        self.pattern_switches += 1
    
    def _update_time_series(self):
        """Update time series data"""
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds()
        
        # Calculate current metrics
        success_rate = self.operations_success / max(1, self.operations_total)
        response_time = sum(self.response_times) / max(1, len(self.response_times))
        ops_per_second = self.operations_total / max(1, elapsed)
        
        # Update peak operations per second
        self.peak_ops_per_second = max(self.peak_ops_per_second, ops_per_second)
        
        # Record time series data
        self.time_series["timestamps"].append(elapsed)
        self.time_series["success_rate"].append(success_rate)
        self.time_series["response_time"].append(response_time)
        self.time_series["operations_per_second"].append(ops_per_second)
    
    def _calculate_final_metrics(self):
        """Calculate final metrics"""
        if not self.end_time:
            self.end_time = datetime.now()
            
        # Calculate success rate
        self.success_rate = self.operations_success / max(1, self.operations_total)
        
        # Calculate response time metrics
        if self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
            
            # Calculate 95th percentile
            sorted_times = sorted(self.response_times)
            idx = int(len(sorted_times) * 0.95)
            self.percentile_95 = sorted_times[idx]
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() if self.start_time else 0
    
    def get_operations_per_second(self) -> float:
        """Get operations per second"""
        elapsed = self.get_elapsed_time()
        return self.operations_total / max(1, elapsed)
    
    def get_layer_distribution(self) -> Dict[OSILayer, float]:
        """Get distribution of operations across layers"""
        result = {}
        for layer, stats in self.layer_stats.items():
            result[layer] = stats["total"] / max(1, self.operations_total)
        return result
    
    def get_success_rate_by_layer(self) -> Dict[OSILayer, float]:
        """Get success rate by layer"""
        result = {}
        for layer, stats in self.layer_stats.items():
            if stats["total"] > 0:
                result[layer] = stats["success"] / stats["total"]
            else:
                result[layer] = 0
        return result
    
    def get_status_code_distribution(self) -> Dict[int, float]:
        """Get distribution of status codes"""
        total = sum(self.status_codes.values())
        return {code: count / max(1, total) for code, count in self.status_codes.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of statistics"""
        return {
            "duration": self.get_elapsed_time(),
            "operations_total": self.operations_total,
            "operations_success": self.operations_success,
            "operations_failed": self.operations_failed,
            "operations_timeout": self.operations_timeout,
            "operations_blocked": self.operations_blocked,
            "success_rate": self.success_rate,
            "ops_per_second": self.get_operations_per_second(),
            "peak_ops_per_second": self.peak_ops_per_second,
            "avg_response_time": self.avg_response_time,
            "percentile_95": self.percentile_95,
            "adaptations": self.adaptations,
            "pattern_switches": self.pattern_switches
        }

class AdaptiveController:
    """Implements adaptive behavior for traffic generation"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.stats = SimulationStats()
        self.current_pattern = config.traffic_pattern
        self.current_intensity = config.intensity
        self.layer_weights = config.layer_distribution.copy()
        
        self.min_success_threshold = 0.3
        self.max_response_threshold = 5000  # ms
        
        self.adaptation_count = 0
        self.pattern_switches = 0
        
        # Learning parameters
        self.learning_rate = config.learning_rate
        self.exploration_rate = 0.1
        
        # Performance tracking for each pattern
        self.pattern_performance = {pattern: {
            "success_rate": 0.5,  # Initial estimate
            "response_time": 1000,  # Initial estimate
            "samples": 0
        } for pattern in TrafficPattern}
        
    def should_adapt(self, current_success_rate: float, 
                   avg_response_time: float) -> bool:
        """Determine if adaptation is needed"""
        if not self.config.adaptation_enabled:
            return False
            
        # Adapt if success rate is too low
        if current_success_rate < self.min_success_threshold:
            return True
            
        # Adapt if response time is too high
        if avg_response_time > self.max_response_threshold:
            return True
            
        # Occasionally explore other patterns
        if random.random() < self.exploration_rate:
            return True
            
        return False
    
    def adapt(self, current_success_rate: float, 
            avg_response_time: float, 
            layer_success_rates: Dict[OSILayer, float]) -> Dict[str, Any]:
        """
        Adapt behavior based on observed performance
        
        Returns dict with changes made
        """
        changes = {}
        
        # Update pattern performance
        if self.pattern_performance[self.current_pattern]["samples"] > 0:
            # Weighted update of performance metrics
            pattern_stats = self.pattern_performance[self.current_pattern]
            weight = min(0.8, 10 / pattern_stats["samples"])  # More weight for fewer samples
            
            pattern_stats["success_rate"] = (
                (1 - weight) * pattern_stats["success_rate"] + 
                weight * current_success_rate
            )
            
            pattern_stats["response_time"] = (
                (1 - weight) * pattern_stats["response_time"] + 
                weight * avg_response_time
            )
            
        self.pattern_performance[self.current_pattern]["samples"] += 1
        
        # Decide if pattern should be changed
        if self.should_adapt(current_success_rate, avg_response_time):
            # Select new pattern based on performance
            old_pattern = self.current_pattern
            patterns = list(TrafficPattern)
            
            # Calculate scores
            pattern_scores = {}
            for pattern in patterns:
                perf = self.pattern_performance[pattern]
                
                # Score based on success rate and response time
                success_score = perf["success_rate"]
                response_score = max(0, 1 - (perf["response_time"] / self.max_response_threshold))
                
                # Combined score
                pattern_scores[pattern] = 0.7 * success_score + 0.3 * response_score
                
                # Apply exploration bonus for less tested patterns
                if perf["samples"] < 5:
                    pattern_scores[pattern] += 0.2
            
            # Select best pattern, occasionally use exploration
            if random.random() < self.exploration_rate:
                # Random exploration
                self.current_pattern = random.choice(patterns)
            else:
                # Select best pattern
                self.current_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
            
            if self.current_pattern != old_pattern:
                changes["pattern"] = {
                    "old": old_pattern.name,
                    "new": self.current_pattern.name
                }
                self.pattern_switches += 1
                self.stats.record_pattern_switch()
        
        # Adjust intensity based on response time
        old_intensity = self.current_intensity
        
        if avg_response_time > self.max_response_threshold:
            # Reduce intensity if response time is too high
            self.current_intensity = max(0.1, self.current_intensity - self.learning_rate)
        elif avg_response_time < self.max_response_threshold * 0.5 and current_success_rate > 0.8:
            # Increase intensity if performance is good
            self.current_intensity = min(1.0, self.current_intensity + self.learning_rate * 0.5)
        
        if abs(self.current_intensity - old_intensity) > 0.01:
            changes["intensity"] = {
                "old": old_intensity,
                "new": self.current_intensity
            }
        
        # Adjust layer weights based on success rates
        old_weights = self.layer_weights.copy()
        
        for layer, success_rate in layer_success_rates.items():
            # Adjust weight based on success rate relative to average
            avg_success = current_success_rate
            relative_success = success_rate - avg_success
            
            # Apply adjustment
            adjustment = self.learning_rate * relative_success
            self.layer_weights[layer] = max(0.05, min(0.5, self.layer_weights[layer] + adjustment))
        
        # Normalize weights
        total = sum(self.layer_weights.values())
        for layer in self.layer_weights:
            self.layer_weights[layer] /= total
        
        # Check if weights changed significantly
        weight_changed = any(abs(self.layer_weights[layer] - old_weights[layer]) > 0.05 
                          for layer in self.layer_weights)
        
        if weight_changed:
            changes["layer_weights"] = {
                "old": {l.name: w for l, w in old_weights.items()},
                "new": {l.name: w for l, w in self.layer_weights.items()}
            }
        
        # Record adaptation if any changes were made
        if changes:
            self.adaptation_count += 1
            self.stats.record_adaptation()
            
        return changes

class TerminalUI:
    """Handles terminal user interface for the simulation"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.stats = None
        self.stdscr = None
        self.running = False
        self.start_time = None
        self.progress = 0.0
        self.status_message = ""
        self.last_update = 0
        
        self.log_messages = deque(maxlen=10)
        self.latest_operations = deque(maxlen=5)
        
        # Colors
        self.colors = {
            "normal": 1,
            "highlight": 2,
            "success": 3,
            "error": 4,
            "warning": 5,
            "info": 6
        }
    
    def setup(self, stats: SimulationStats):
        """Set up the UI with statistics object"""
        self.stats = stats
    
    def start(self):
        """Start the UI in a separate thread"""
        self.running = True
        thread = threading.Thread(target=self._ui_thread)
        thread.daemon = True
        thread.start()
    
    def stop(self):
        """Stop the UI"""
        self.running = False
    
    def _ui_thread(self):
        """Main UI thread"""
        try:
            wrapper(self._run_ui)
        except Exception as e:
            logger.error(f"UI error: {str(e)}", exc_info=True)
    
    def _run_ui(self, stdscr):
        """Run the UI with curses"""
        self.stdscr = stdscr
        self.start_time = time.time()
        
        # Set up colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, -1)  # normal
        curses.init_pair(2, curses.COLOR_CYAN, -1)   # highlight
        curses.init_pair(3, curses.COLOR_GREEN, -1)  # success
        curses.init_pair(4, curses.COLOR_RED, -1)    # error
        curses.init_pair(5, curses.COLOR_YELLOW, -1) # warning
        curses.init_pair(6, curses.COLOR_BLUE, -1)   # info
        
        # Hide cursor
        curses.curs_set(0)
        
        # Main loop
        while self.running:
            try:
                self._update_ui()
                time.sleep(0.1)
            except Exception as e:
                self.log_messages.append(f"UI Error: {str(e)}")
    
    def _update_ui(self):
        """Update the UI"""
        if time.time() - self.last_update < 0.2:
            return
            
        self.last_update = time.time()
        self.stdscr.clear()
        
        # Get terminal size
        height, width = self.stdscr.getmaxyx()
        
        # Draw header
        self._draw_header(0, 0, width)
        
        # Draw progress bar
        self._draw_progress_bar(2, 0, width)
        
        # Draw statistics
        self._draw_statistics(4, 0, width)
        
        # Draw latest operations
        self._draw_operations(11, 0, width)
        
        # Draw log messages
        self._draw_logs(height - 12, 0, width)
        
        # Draw status bar
        self._draw_status_bar(height - 1, 0, width)
        
        self.stdscr.refresh()
    
    def _draw_header(self, y, x, width):
        """Draw the header"""
        title = "Network Simulation Framework"
        mode = f"Mode: {self.config.mode.name}"
        target = f"Target: {self.config.target}"
        
        self.stdscr.addstr(y, (width - len(title)) // 2, title, curses.color_pair(self.colors["highlight"]) | curses.A_BOLD)
        self.stdscr.addstr(y, 0, mode, curses.color_pair(self.colors["normal"]))
        self.stdscr.addstr(y, width - len(target) - 1, target, curses.color_pair(self.colors["normal"]))
    
    def _draw_progress_bar(self, y, x, width):
        """Draw the progress bar"""
        elapsed = time.time() - self.start_time
        duration = self.config.duration
        
        if duration > 0:
            progress = min(1.0, elapsed / duration)
            self.progress = progress
        else:
            progress = self.progress
            
        bar_width = width - 20
        filled_width = int(bar_width * progress)
        
        time_text = f"{int(elapsed)}s / {duration}s"
        
        self.stdscr.addstr(y, x, "Progress: [", curses.color_pair(self.colors["normal"]))
        
        # Draw filled part
        for i in range(filled_width):
            self.stdscr.addstr(y, x + 10 + i, "█", curses.color_pair(self.colors["highlight"]))
        
        # Draw empty part
        for i in range(filled_width, bar_width):
            self.stdscr.addstr(y, x + 10 + i, "░", curses.color_pair(self.colors["normal"]))
            
        self.stdscr.addstr(y, x + 10 + bar_width, f"] {time_text}", curses.color_pair(self.colors["normal"]))
    
    def _draw_statistics(self, y, x, width):
        """Draw statistics"""
        if not self.stats:
            return
            
        # Basic stats
        total = self.stats.operations_total
        success = self.stats.operations_success
        failed = self.stats.operations_failed
        timeout = self.stats.operations_timeout
        blocked = self.stats.operations_blocked
        
        success_rate = success / max(1, total) * 100
        ops_per_second = self.stats.get_operations_per_second()
        
        avg_response = 0
        if self.stats.response_times:
            avg_response = sum(self.stats.response_times) / len(self.stats.response_times)
            
        # Draw stats
        self.stdscr.addstr(y, x, "Statistics:", curses.color_pair(self.colors["highlight"]) | curses.A_BOLD)
        
        self.stdscr.addstr(y + 1, x, f"Operations: {total} total, ", curses.color_pair(self.colors["normal"]))
        self.stdscr.addstr(f"{success} success", curses.color_pair(self.colors["success"]))
        self.stdscr.addstr(", ", curses.color_pair(self.colors["normal"]))
        self.stdscr.addstr(f"{failed} failed", curses.color_pair(self.colors["error"]))
        self.stdscr.addstr(", ", curses.color_pair(self.colors["normal"]))
        self.stdscr.addstr(f"{timeout} timeout", curses.color_pair(self.colors["warning"]))
        self.stdscr.addstr(", ", curses.color_pair(self.colors["normal"]))
        self.stdscr.addstr(f"{blocked} blocked", curses.color_pair(self.colors["info"]))
        
        self.stdscr.addstr(y + 2, x, f"Success Rate: {success_rate:.1f}%   Ops/s: {ops_per_second:.2f}   Avg Response: {avg_response:.2f}ms", 
                        curses.color_pair(self.colors["normal"]))
        
        # Layer distribution
        if self.stats.layer_stats:
            self.stdscr.addstr(y + 4, x, "Layer Distribution:", curses.color_pair(self.colors["highlight"]))
            
            layer_items = []
            for layer, stats in self.stats.layer_stats.items():
                if stats["total"] > 0:
                    layer_success = stats["success"] / stats["total"] * 100
                    layer_items.append(f"{layer.name}: {stats['total']} ops ({layer_success:.1f}% success)")
            
            if layer_items:
                self.stdscr.addstr(y + 5, x, "  ".join(layer_items), curses.color_pair(self.colors["normal"]))
        
        # Adaptation info
        if hasattr(self, 'adapter') and self.adapter:
            pattern = self.adapter.current_pattern.name
            intensity = self.adapter.current_intensity
            adaptations = self.adapter.adaptation_count
            
            self.stdscr.addstr(y + 7, x, f"Traffic Pattern: {pattern}   Intensity: {intensity:.2f}   Adaptations: {adaptations}", 
                            curses.color_pair(self.colors["normal"]))
    
    def _draw_operations(self, y, x, width):
        """Draw latest operations"""
        self.stdscr.addstr(y, x, "Latest Operations:", curses.color_pair(self.colors["highlight"]) | curses.A_BOLD)
        
        for i, op in enumerate(self.latest_operations):
            if i >= 5:  # Limit to 5 operations
                break
                
            op_text = f"{op['id']}: {op['layer']} {op['status']} ({op['time']:.2f}ms)"
            if op['code']:
                op_text += f" [Status: {op['code']}]"
                
            color = self.colors["normal"]
            if op['status'] == "SUCCESS":
                color = self.colors["success"]
            elif op['status'] == "FAILED":
                color = self.colors["error"]
            elif op['status'] == "TIMEOUT":
                color = self.colors["warning"]
            elif op['status'] == "BLOCKED":
                color = self.colors["info"]
                
            self.stdscr.addstr(y + i + 1, x, op_text, curses.color_pair(color))
    
    def _draw_logs(self, y, x, width):
        """Draw log messages"""
        self.stdscr.addstr(y, x, "Log Messages:", curses.color_pair(self.colors["highlight"]) | curses.A_BOLD)
        
        for i, msg in enumerate(self.log_messages):
            if i >= 10:  # Limit to 10 messages
                break
                
            # Truncate message if too long
            if len(msg) > width - 2:
                msg = msg[:width - 5] + "..."
                
            self.stdscr.addstr(y + i + 1, x, msg, curses.color_pair(self.colors["normal"]))
    
    def _draw_status_bar(self, y, x, width):
        """Draw the status bar"""
        status = self.status_message or "Running simulation..."
        
        # Truncate if too long
        if len(status) > width - 2:
            status = status[:width - 5] + "..."
            
        self.stdscr.addstr(y, x, status, curses.color_pair(self.colors["highlight"]) | curses.A_REVERSE)
    
    def log(self, message: str):
        """Add a log message"""
        self.log_messages.appendleft(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    
    def update_status(self, message: str):
        """Update the status message"""
        self.status_message = message
    
    def record_operation(self, operation: NetworkOperation):
        """Record an operation for display"""
        if not operation.result:
            return
            
        self.latest_operations.appendleft({
            "id": operation.id,
            "layer": operation.layer.name,
            "status": operation.result.status.name,
            "time": operation.result.response_time,
            "code": operation.result.status_code
        })

class NetworkSimulator:
    """Main simulator class"""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.stats = SimulationStats()
        self.traffic_generator = TrafficGenerator(config)
        self.network_operator = NetworkOperator(config)
        self.ui = TerminalUI(config)
        
        if config.adaptation_enabled:
            self.adapter = AdaptiveController(config)
            self.adapter.stats = self.stats
            self.ui.adapter = self.adapter
        else:
            self.adapter = None
            
        self.operation_queue = queue.Queue()
        self.stop_flag = threading.Event()
        
        self.total_operations = 0
        self.completed_operations = 0
        
        self.worker_threads = []
        
        # Set up UI
        self.ui.setup(self.stats)
    
    def start(self):
        """Start the simulation"""
        logger.info(f"Starting simulation against {self.config.target}")
        logger.info(f"Mode: {self.config.mode.name}, Duration: {self.config.duration}s, Threads: {self.config.threads}")
        
        # Start statistics
        self.stats.start()
        
        # Start UI
        self.ui.start()
        self.ui.log(f"Starting simulation against {self.config.target}")
        self.ui.update_status("Generating traffic pattern...")
        
        # Generate traffic pattern
        pattern = self.config.traffic_pattern
        duration = self.config.duration
        intensity = self.config.intensity
        
        if self.adapter:
            pattern = self.adapter.current_pattern
            intensity = self.adapter.current_intensity
            
        timings = self.traffic_generator.generate_timings(pattern, duration, intensity)
        self.total_operations = len(timings)
        
        self.ui.log(f"Generated {self.total_operations} operations with pattern {pattern.name}")
        self.ui.update_status(f"Starting operation execution with {self.config.threads} threads...")
        
        # Create worker threads
        for i in range(self.config.threads):
            thread = threading.Thread(target=self._worker_thread, args=(i,))
            thread.daemon = True
            self.worker_threads.append(thread)
            thread.start()
            
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._scheduler_thread, args=(timings,))
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Start adaptation thread if enabled
        if self.adapter:
            adaptation_thread = threading.Thread(target=self._adaptation_thread)
            adaptation_thread.daemon = True
            adaptation_thread.start()
        
        # Wait for completion
        try:
            scheduler_thread.join()
            self.operation_queue.join()
        except KeyboardInterrupt:
            self.ui.log("Interrupted by user")
            self.stop()
        
        # Stop the simulation
        self.stop()
        
        # Return results
        return self.stats.get_summary()
    
    def stop(self):
        """Stop the simulation"""
        if self.stop_flag.is_set():
            return
            
        self.stop_flag.set()
        self.ui.update_status("Stopping simulation...")
        self.ui.log("Stopping simulation...")
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=1.0)
            
        # Clean up resources
        self.network_operator.cleanup()
        
        # Stop statistics
        self.stats.stop()
        
        # Final UI update
        self.ui.update_status(f"Simulation complete. {self.stats.operations_total} operations executed.")
        time.sleep(1.0)  # Let the UI update
        
        # Stop UI
        self.ui.stop()
    
    def _scheduler_thread(self, timings):
        """Thread for scheduling operations based on timings"""
        start_time = time.time()
        
        for i, timestamp in enumerate(timings):
            # Check if we should stop
            if self.stop_flag.is_set():
                break
                
            # Wait until scheduled time
            wait_time = timestamp - (time.time() - start_time)
            if wait_time > 0:
                time.sleep(wait_time)
                
            # Create operation
            layer = self.traffic_generator.select_layer()
            op_id = f"op-{i+1}"
            
            # Generate parameters
            params = self.traffic_generator.generate_operation_parameters(
                layer, self.config.target
            )
            
            # Create operation
            operation = NetworkOperation(
                layer=layer,
                operation_id=op_id,
                target_info={
                    "host": self.config.target.host,
                    "ip_address": self.config.target.ip_address,
                    "port": self.config.target.port,
                    "protocol": self.config.target.protocol,
                    "path": self.config.target.path,
                    "ssl_enabled": self.config.target.ssl_enabled
                },
                parameters=params
            )
            
            # Add to queue
            self.operation_queue.put(operation)
            
        # Signal end of operations
        self.ui.log(f"Scheduled {len(timings)} operations")
    
    def _worker_thread(self, worker_id):
        """Worker thread for executing operations"""
        self.ui.log(f"Started worker thread {worker_id}")
        
        while not self.stop_flag.is_set():
            try:
                # Get operation from queue with timeout
                try:
                    operation = self.operation_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                # Execute operation
                try:
                    result = self.network_operator.execute_operation(operation)
                    operation.result = result
                    
                    # Record statistics
                    self.stats.record_operation(operation)
                    self.config.target.update_stats(result)
                    
                    # Update UI
                    self.ui.record_operation(operation)
                    
                    # Update progress
                    self.completed_operations += 1
                    if self.total_operations > 0:
                        progress = self.completed_operations / self.total_operations
                        self.ui.progress = progress
                except Exception as e:
                    self.ui.log(f"Error executing operation {operation.id}: {str(e)}")
                    logger.error(f"Operation error: {str(e)}", exc_info=True)
                
                # Mark as done
                self.operation_queue.task_done()
                
            except Exception as e:
                self.ui.log(f"Worker error: {str(e)}")
                logger.error(f"Worker error: {str(e)}", exc_info=True)
        
        self.ui.log(f"Worker thread {worker_id} stopped")
    
    def _adaptation_thread(self):
        """Thread for adaptive behavior"""
        if not self.adapter:
            return
            
        self.ui.log("Started adaptation thread")
        adaptation_interval = 5.0  # seconds
        
        last_adaptation = time.time()
        
        while not self.stop_flag.is_set():
            time.sleep(0.5)
            
            # Check if it's time to adapt
            if time.time() - last_adaptation < adaptation_interval:
                continue
                
            # Get current stats
            current_success_rate = self.stats.operations_success / max(1, self.stats.operations_total)
            avg_response_time = sum(self.stats.response_times) / max(1, len(self.stats.response_times)) if self.stats.response_times else 0
            
            # Get layer success rates
            layer_success_rates = self.stats.get_success_rate_by_layer()
            
            # Adapt behavior
            changes = self.adapter.adapt(
                current_success_rate=current_success_rate,
                avg_response_time=avg_response_time,
                layer_success_rates=layer_success_rates
            )
            
            # Log changes
            if changes:
                self.ui.log(f"Adapted behavior: {json.dumps(changes)}")
                
                # Update config
                self.config.traffic_pattern = self.adapter.current_pattern
                self.config.intensity = self.adapter.current_intensity
                self.config.layer_distribution = self.adapter.layer_weights
                
            last_adaptation = time.time()
        
        self.ui.log("Adaptation thread stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Network Simulation Framework for stress-testing infrastructure"
    )
    
    # Target configuration
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Target host or URL (e.g., example.com or https://example.com/path)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Target port (default: 80 for HTTP, 443 for HTTPS)"
    )
    
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["http", "https", "dns"],
        help="Protocol to use (default: derived from URL)"
    )
    
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Use SSL/TLS (default: true for HTTPS)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=[m.name.lower() for m in SimulationMode],
        default="analyze",
        help="Simulation mode (default: analyze)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="Simulation duration in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--threads", "-n",
        type=int,
        default=10,
        help="Number of worker threads (default: 10)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Operation timeout in seconds (default: 10.0)"
    )
    
    # Traffic configuration
    parser.add_argument(
        "--pattern",
        type=str,
        choices=[p.name.lower() for p in TrafficPattern],
        default="adaptive",
        help="Traffic pattern (default: adaptive)"
    )
    
    parser.add_argument(
        "--intensity", "-i",
        type=float,
        default=0.5,
        help="Traffic intensity, 0.0-1.0 (default: 0.5)"
    )
    
    # Authentication
    parser.add_argument(
        "--username", "-u",
        type=str,
        help="Authentication username"
    )
    
    parser.add_argument(
        "--password", "-pw",
        type=str,
        help="Authentication password"
    )
    
    # SSL/TLS options
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable SSL certificate verification"
    )
    
    # Other options
    parser.add_argument(
        "--no-adapt",
        action="store_true",
        help="Disable adaptive behavior"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create configuration
    config = SimulationConfig().from_args(args)
    
    # Validate configuration
    valid, message = config.validate()
    if not valid:
        print(f"Error: {message}")
        return 1
    
    # Create simulator
    simulator = NetworkSimulator(config)
    
    # Run simulation
    try:
        results = simulator.start()
        
        # Print results
        print("\nSimulation Results:")
        print("=" * 50)
        print(f"Target: {config.target}")
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Operations: {results['operations_total']} total, {results['operations_success']} successful")
        print(f"Success Rate: {results['success_rate'] * 100:.2f}%")
        print(f"Operations/sec: {results['ops_per_second']:.2f} (peak: {results['peak_ops_per_second']:.2f})")
        print(f"Avg Response Time: {results['avg_response_time']:.2f}ms")
        print(f"95th Percentile: {results['percentile_95']:.2f}ms")
        
        if config.adaptation_enabled:
            print(f"Adaptations: {results['adaptations']}")
            print(f"Pattern Switches: {results['pattern_switches']}")
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        simulator.stop()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())