"""
Parameters for parafoil dynamics simulation.

All aerodynamic coefficients follow the naming convention from the project requirements:
- Lift: c_L0, c_La, c_Lds
- Drag: c_D0, c_Da2, c_Dds  
- Side force: c_Yb
- Moments/Damping: c_lp, c_lda, c_m0, c_ma, c_mq, c_mds, c_nr, c_nda
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import yaml


@dataclass
class Params:
    """
    Parafoil simulation parameters.
    
    Physical Constants:
        rho: Air density [kg/m^3]
        g: Gravitational acceleration [m/s^2]
    
    Mass Properties:
        m: Total mass (canopy + payload) [kg]
        I_B: Inertia tensor in body frame [kg*m^2], shape (3,3)
    
    Geometry:
        S: Canopy reference area [m^2]
        b: Canopy span [m]
        c: Canopy chord [m]
        S_pd: Payload drag area [m^2]
        r_pd_B: Payload position in body frame [m]
    
    Aerodynamic Coefficients (Lift/Drag):
        c_L0: Lift coefficient at zero alpha
        c_La: Lift curve slope (dC_L/dalpha)
        c_Lds: Lift coefficient due to symmetric brake
        c_D0: Parasite drag coefficient
        c_Da2: Induced drag factor (C_D = c_D0 + c_Da2 * alpha^2)
        c_Dds: Drag coefficient due to symmetric brake
    
    Aerodynamic Coefficients (Side Force):
        c_Yb: Side force due to sideslip (dC_Y/dbeta)
    
    Aerodynamic Coefficients (Moments):
        c_lp: Roll damping derivative
        c_lda: Roll due to asymmetric brake
        c_m0: Pitching moment at zero alpha
        c_ma: Pitching moment slope (dC_m/dalpha)
        c_mq: Pitch damping derivative
        c_mds: Pitching moment due to symmetric brake (flare effect)
        c_nr: Yaw damping derivative
        c_nda: Yaw due to asymmetric brake
    
    Actuator Dynamics:
        tau_act: Actuator time constant [s]
    
    Numerical:
        eps: Small value for division protection
    """
    
    # Physical constants (from reference model)
    rho: float = 1.29   # Air density [kg/m^3] - matches reference
    g: float = 9.81     # Gravitational acceleration [m/s^2]
    
    # =================================================================
    # Mass properties - Two-body model (canopy + payload)
    # =================================================================
    # Canopy: 200g, receives aerodynamic forces and control moments
    # Payload: 2000g, suspended below canopy, provides pendulum stability
    m_canopy: float = 0.2       # Canopy mass [kg]
    m_payload: float = 2.0      # Payload mass [kg]
    m: float = 2.2              # Total mass [kg] = m_canopy + m_payload
    
    # Canopy inertia (what matters for yaw response to control inputs)
    # Small canopy inertia allows fast rotation (~360 deg/s at 100% brake)
    # I_canopy ≈ m_canopy * (b/2)^2 for yaw
    I_canopy: np.ndarray = field(default_factory=lambda: np.diag([0.047, 0.023, 0.047]))
    
    # Effective system inertia (for slower modes like roll/pitch damping)
    # This includes the apparent inertia from payload on pendulum
    I_B: np.ndarray = field(default_factory=lambda: np.diag([0.8, 0.15, 0.85]))
    
    # Line length from canopy to payload (pendulum arm)
    line_length: float = 0.5    # [m] suspension line length
    
    # Geometry (reference model dimensions)
    S: float = 1.5      # Canopy reference area [m^2]
    b: float = 1.88     # Canopy span [m]
    c: float = 0.80     # Canopy chord [m]
    S_pd: float = 0.1   # Payload drag area [m^2]
    c_D_pd: float = 1.0 # Payload drag coefficient
    
    # Position vectors in body frame (origin at system CG)
    # r_canopy_B: canopy aerodynamic center position (above CG, negative z in NED)
    # r_pd_B: payload position (below CG, positive z in NED)
    r_canopy_B: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.3]))  # Canopy 0.3m above CG
    r_pd_B: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2]))       # Payload 0.2m below CG
    
    # Pendulum stiffness for roll/pitch stability (simplified model)
    # This represents the restoring moment due to payload weight offset
    # M_pendulum = -k_pendulum * [roll, pitch, 0] in body frame
    pendulum_arm: float = 0.3   # Effective pendulum arm length [m] (small parafoil)
    
    # Lift coefficients - TUNED to match flight log 09_01_06
    # Target: V=4.3 m/s, sink=0.85 m/s, L/D=5.0 at zero brake
    c_L0: float = 0.55      # Zero-alpha lift coefficient
    c_La: float = 3.80      # Lift curve slope [1/rad]
    c_Lds: float = 0.20     # Symmetric brake effect on lift

    # Drag coefficients - TUNED to match flight log 09_01_06
    c_D0: float = 0.16      # Parasite drag
    c_Da2: float = 0.50     # Induced drag factor
    c_Dds: float = 0.60     # Symmetric brake effect on drag

    # Stall model parameters
    # Parafoil stalls at high alpha, causing lift drop and drag spike
    alpha_stall: float = 0.28           # Stall angle [rad] (~16 deg) at zero brake
    alpha_stall_brake: float = 0.06     # Stall angle reduction per unit brake [rad]
    alpha_stall_width: float = 0.08     # Wider transition for gradual stall
    c_D_stall: float = 0.40             # Additional drag in full stall

    # Side force coefficients
    # CRITICAL: Must be large enough for coordinated turning
    # Coordinated turn: C_Y * q * S = m * V * r (side force = centripetal force)
    # For 75 deg/s at 5° sideslip: c_Yb ≈ -6.8
    c_Yb: float = -6.8      # Side force due to sideslip [1/rad] (tuned for coordination)
    
    # Roll moment coefficients (from reference model)
    c_lp: float = -0.84     # Roll damping derivative
    c_lda: float = -0.005   # Roll due to asymmetric brake
    c_lb: float = 0.0       # Sideslip-roll coupling (disabled)
    
    # Pitch moment coefficients (from reference model)
    c_m0: float = 0.1       # Zero-alpha pitching moment
    c_ma: float = -0.72     # Pitching moment slope [1/rad]
    c_mq: float = -1.49     # Pitch damping derivative
    c_mds: float = 0.0      # Symmetric brake -> pitch moment (flare/zoom). Default off; tune per airframe.

    # Yaw moment coefficients (from reference model)
    c_nr: float = -0.27     # Yaw damping derivative
    c_nda: float = -0.133   # Yaw due to asymmetric brake (negative: left brake -> turn left)
    c_nb: float = 0.15      # Yaw due to sideslip (wind-yaw coupling, positive: right wind -> turn right)

    # Weathercock effect coefficient
    # Models destabilizing yaw moment from crosswind component relative to heading
    # Positive value: crosswind from right → turn right (away from wind source)
    # This creates unstable equilibrium when flying into headwind
    # Real parafoils naturally turn from headwind to tailwind
    c_n_weath: float = 0.02  # Weathercock coefficient [1/(m/s)]

    # Actuator dynamics
    tau_act: float = 0.2    # Actuator time constant [s]
    
    # Numerical protection
    eps: float = 1e-6       # Small value for division protection
    V_min: float = 1.0      # Minimum velocity for aerodynamic calculations [m/s]
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.I_B = np.asarray(self.I_B, dtype=np.float64)
        self.I_canopy = np.asarray(self.I_canopy, dtype=np.float64)
        self.r_pd_B = np.asarray(self.r_pd_B, dtype=np.float64)
        # Precompute inverse inertia tensors
        self._I_B_inv = np.linalg.inv(self.I_B)
        self._I_canopy_inv = np.linalg.inv(self.I_canopy)
    
    @property
    def I_B_inv(self) -> np.ndarray:
        """Inverse of system inertia tensor (for roll/pitch)."""
        return self._I_B_inv
    
    @property
    def I_canopy_inv(self) -> np.ndarray:
        """Inverse of canopy inertia tensor (for yaw control response)."""
        return self._I_canopy_inv
    
    def copy(self) -> 'Params':
        """Create a copy of parameters."""
        p = Params(
            rho=self.rho, g=self.g, m=self.m,
            I_B=self.I_B.copy(),
            S=self.S, b=self.b, c=self.c,
            S_pd=self.S_pd, c_D_pd=self.c_D_pd,
            r_pd_B=self.r_pd_B.copy(),
            c_L0=self.c_L0, c_La=self.c_La, c_Lds=self.c_Lds,
            c_D0=self.c_D0, c_Da2=self.c_Da2, c_Dds=self.c_Dds,
            c_Yb=self.c_Yb,
            c_lp=self.c_lp, c_lda=self.c_lda, c_lb=self.c_lb,
            c_m0=self.c_m0, c_ma=self.c_ma, c_mq=self.c_mq, c_mds=self.c_mds,
            c_nr=self.c_nr, c_nda=self.c_nda, c_nb=self.c_nb, c_n_weath=self.c_n_weath,
            tau_act=self.tau_act,
            eps=self.eps, V_min=self.V_min
        )
        return p
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Params':
        """Create Params from a dictionary."""
        params = cls()
        
        # Update scalar parameters
        scalar_fields = [
            'rho', 'g', 'm', 'S', 'b', 'c', 'S_pd', 'c_D_pd',
            'c_L0', 'c_La', 'c_Lds',
            'c_D0', 'c_Da2', 'c_Dds',
            'c_Yb',
            'c_lp', 'c_lda', 'c_lb',
            'c_m0', 'c_ma', 'c_mq', 'c_mds',
            'c_nr', 'c_nda', 'c_nb', 'c_n_weath',
            'tau_act', 'eps', 'V_min'
        ]
        
        for field_name in scalar_fields:
            if field_name in d:
                setattr(params, field_name, float(d[field_name]))
        
        # Update array parameters
        if 'I_B' in d:
            params.I_B = np.asarray(d['I_B'], dtype=np.float64)
            params._I_B_inv = np.linalg.inv(params.I_B)
        if 'r_pd_B' in d:
            params.r_pd_B = np.asarray(d['r_pd_B'], dtype=np.float64)
        
        return params
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Params':
        """Load parameters from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle nested structure (ROS2 style)
        if 'parafoil_simulator' in data:
            data = data['parafoil_simulator']
        if 'ros__parameters' in data:
            data = data['ros__parameters']
        
        return cls.from_dict(data)
    
    @classmethod
    def from_ros_params(cls, node) -> 'Params':
        """
        Load parameters from a ROS2 node.
        
        Args:
            node: A ROS2 node with declared parameters
            
        Returns:
            Params object with values from ROS parameters
        """
        params = cls()
        
        # Define parameter mappings
        scalar_params = [
            ('rho', 1.225), ('g', 9.81), ('m', 250.0),
            ('S', 30.0), ('b', 10.0), ('c', 3.0),
            ('S_pd', 0.5), ('c_D_pd', 1.0),
            ('c_L0', 0.4), ('c_La', 3.5), ('c_Lds', -0.3),
            ('c_D0', 0.1), ('c_Da2', 0.5), ('c_Dds', 0.4),
            ('c_Yb', -0.3),
            ('c_lp', -0.4), ('c_lda', 0.1),
            ('c_m0', 0.02), ('c_ma', -0.5), ('c_mq', -2.0), ('c_mds', 0.0),
            ('c_nr', -0.3), ('c_nda', -0.05),
            ('tau_act', 0.2), ('eps', 1e-6), ('V_min', 1.0)
        ]
        
        for name, default in scalar_params:
            node.declare_parameter(name, default)
            setattr(params, name, node.get_parameter(name).value)
        
        # Array parameters
        node.declare_parameter('I_B_diag', [100.0, 50.0, 120.0])
        I_diag = node.get_parameter('I_B_diag').value
        params.I_B = np.diag(I_diag)
        params._I_B_inv = np.linalg.inv(params.I_B)
        
        node.declare_parameter('r_pd_B', [0.0, 0.0, 5.0])
        params.r_pd_B = np.array(node.get_parameter('r_pd_B').value)
        
        return params
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'rho': self.rho, 'g': self.g, 'm': self.m,
            'I_B': self.I_B.tolist(),
            'S': self.S, 'b': self.b, 'c': self.c,
            'S_pd': self.S_pd, 'c_D_pd': self.c_D_pd,
            'r_pd_B': self.r_pd_B.tolist(),
            'c_L0': self.c_L0, 'c_La': self.c_La, 'c_Lds': self.c_Lds,
            'c_D0': self.c_D0, 'c_Da2': self.c_Da2, 'c_Dds': self.c_Dds,
            'c_Yb': self.c_Yb,
            'c_lp': self.c_lp, 'c_lda': self.c_lda, 'c_lb': self.c_lb,
            'c_m0': self.c_m0, 'c_ma': self.c_ma, 'c_mq': self.c_mq, 'c_mds': self.c_mds,
            'c_nr': self.c_nr, 'c_nda': self.c_nda, 'c_nb': self.c_nb, 'c_n_weath': self.c_n_weath,
            'tau_act': self.tau_act,
            'eps': self.eps, 'V_min': self.V_min
        }
