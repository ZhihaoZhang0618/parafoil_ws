"""
Unit tests for parafoil dynamics library.

Test categories:
1. Quaternion normalization: Verify quaternion stays normalized during integration
2. Substep convergence: Smaller dt_max should give more accurate results
3. Finite values: No NaN/Inf under reasonable conditions
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

# Import modules to test
from parafoil_dynamics.state import State, StateDot, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.math3d import (
    normalize_quaternion, quat_multiply, quat_to_rotmat, rotmat_to_quat,
    quat_from_euler, quat_to_euler, quat_derivative, compute_aero_angles
)
from parafoil_dynamics.dynamics import dynamics, get_body_acceleration
from parafoil_dynamics.integrators import (
    euler_step, semi_implicit_step, rk4_step,
    integrate_with_substeps, IntegratorType
)
from parafoil_dynamics.wind import WindModel, WindConfig
from parafoil_dynamics.sensors import SensorModel, SensorConfig


class TestQuaternionOperations:
    """Test quaternion math utilities."""
    
    def test_normalize_quaternion(self):
        """Test quaternion normalization."""
        q = np.array([1.0, 2.0, 3.0, 4.0])
        q_norm = normalize_quaternion(q)
        assert_allclose(np.linalg.norm(q_norm), 1.0, atol=1e-10)
    
    def test_normalize_small_quaternion(self):
        """Test normalization of very small quaternion returns identity."""
        q = np.array([1e-15, 0.0, 0.0, 0.0])
        q_norm = normalize_quaternion(q)
        assert_allclose(q_norm, [1.0, 0.0, 0.0, 0.0], atol=1e-10)
    
    def test_quat_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q = normalize_quaternion(np.array([1.0, 2.0, 3.0, 4.0]))
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        
        result = quat_multiply(q, q_id)
        assert_allclose(result, q, atol=1e-10)
        
        result = quat_multiply(q_id, q)
        assert_allclose(result, q, atol=1e-10)
    
    def test_quat_to_rotmat_identity(self):
        """Test identity quaternion gives identity rotation matrix."""
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rotmat(q_id)
        assert_allclose(R, np.eye(3), atol=1e-10)
    
    def test_rotmat_quat_roundtrip(self):
        """Test quaternion -> rotmat -> quaternion roundtrip."""
        q = normalize_quaternion(np.array([0.5, 0.5, 0.5, 0.5]))
        R = quat_to_rotmat(q)
        q_back = rotmat_to_quat(R)
        
        # Quaternions may differ by sign
        if np.dot(q, q_back) < 0:
            q_back = -q_back
        assert_allclose(q, q_back, atol=1e-10)
    
    def test_euler_quat_roundtrip(self):
        """Test Euler -> quaternion -> Euler roundtrip."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        q = quat_from_euler(roll, pitch, yaw)
        roll2, pitch2, yaw2 = quat_to_euler(q)
        
        assert_allclose([roll, pitch, yaw], [roll2, pitch2, yaw2], atol=1e-10)


class TestQuaternionNormalizationDuringIntegration:
    """Test that quaternion stays normalized during integration."""
    
    @pytest.fixture
    def initial_state(self):
        """Create initial state at altitude."""
        return State(
            p_I=np.array([0.0, 0.0, -500.0]),  # 500m altitude
            v_I=np.array([10.0, 0.0, 2.0]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.01, 0.02, 0.01]),  # Some rotation
            delta=np.array([0.1, 0.1]),
            t=0.0
        )
    
    @pytest.fixture
    def params(self):
        return Params()
    
    @pytest.fixture
    def cmd(self):
        return ControlCmd(delta_cmd=np.array([0.2, 0.3]))
    
    def test_euler_quaternion_normalization(self, initial_state, params, cmd):
        """Test quaternion stays normalized with Euler integration."""
        state = initial_state.copy()
        dt = 0.01
        
        for _ in range(1000):
            state = euler_step(dynamics, state, cmd, params, dt)
            q_norm = np.linalg.norm(state.q_IB)
            assert_allclose(q_norm, 1.0, atol=1e-6)
    
    def test_rk4_quaternion_normalization(self, initial_state, params, cmd):
        """Test quaternion stays normalized with RK4 integration."""
        state = initial_state.copy()
        dt = 0.01
        
        for _ in range(1000):
            state = rk4_step(dynamics, state, cmd, params, dt)
            q_norm = np.linalg.norm(state.q_IB)
            assert_allclose(q_norm, 1.0, atol=1e-6)
    
    def test_semi_implicit_quaternion_normalization(self, initial_state, params, cmd):
        """Test quaternion stays normalized with semi-implicit integration."""
        state = initial_state.copy()
        dt = 0.01
        
        for _ in range(1000):
            state = semi_implicit_step(dynamics, state, cmd, params, dt)
            q_norm = np.linalg.norm(state.q_IB)
            assert_allclose(q_norm, 1.0, atol=1e-6)


class TestSubstepConvergence:
    """Test that smaller dt_max gives more accurate results."""
    
    @pytest.fixture
    def initial_state(self):
        return State(
            p_I=np.array([0.0, 0.0, -500.0]),
            v_I=np.array([10.0, 0.0, 2.0]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.0, 0.0, 0.0]),
            delta=np.array([0.0, 0.0]),
            t=0.0
        )
    
    @pytest.fixture
    def params(self):
        return Params()
    
    @pytest.fixture
    def cmd(self):
        return ControlCmd(delta_cmd=np.array([0.3, 0.3]))
    
    def test_substep_convergence_euler(self, initial_state, params, cmd):
        """Test that Euler converges with smaller substeps."""
        ctl_dt = 0.1
        
        # Integrate with different dt_max values
        dt_max_values = [0.1, 0.05, 0.025, 0.0125]
        final_states = []
        
        for dt_max in dt_max_values:
            state = integrate_with_substeps(
                dynamics, initial_state.copy(), cmd, params,
                ctl_dt, dt_max, IntegratorType.EULER
            )
            final_states.append(state)
        
        # Check convergence: differences should decrease
        diffs = []
        for i in range(len(final_states) - 1):
            diff = np.linalg.norm(
                final_states[i].p_I - final_states[i+1].p_I
            )
            diffs.append(diff)
        
        # Each difference should be smaller (or at least not much larger)
        for i in range(len(diffs) - 1):
            assert diffs[i+1] <= diffs[i] * 1.1  # Allow 10% tolerance
    
    def test_substep_convergence_rk4(self, initial_state, params, cmd):
        """Test that RK4 converges with smaller substeps."""
        ctl_dt = 0.1
        
        dt_max_values = [0.1, 0.05, 0.025]
        final_states = []
        
        for dt_max in dt_max_values:
            state = integrate_with_substeps(
                dynamics, initial_state.copy(), cmd, params,
                ctl_dt, dt_max, IntegratorType.RK4
            )
            final_states.append(state)
        
        # RK4 should have much smaller differences
        diff_01 = np.linalg.norm(final_states[0].p_I - final_states[1].p_I)
        diff_12 = np.linalg.norm(final_states[1].p_I - final_states[2].p_I)
        
        # RK4 differences should be small
        assert diff_01 < 0.01  # Less than 1cm difference
        assert diff_12 < 0.01


class TestFiniteValues:
    """Test that dynamics produces finite values under various conditions."""
    
    @pytest.fixture
    def params(self):
        return Params()
    
    def test_dynamics_finite_normal_state(self, params):
        """Test dynamics produces finite output for normal state."""
        state = State(
            p_I=np.array([0.0, 0.0, -500.0]),
            v_I=np.array([10.0, 0.0, 2.0]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.1, 0.1, 0.1]),
            delta=np.array([0.5, 0.5]),
            t=0.0
        )
        cmd = ControlCmd(delta_cmd=np.array([0.5, 0.5]))
        
        state_dot = dynamics(state, cmd, params)
        
        assert np.all(np.isfinite(state_dot.p_I_dot))
        assert np.all(np.isfinite(state_dot.v_I_dot))
        assert np.all(np.isfinite(state_dot.q_IB_dot))
        assert np.all(np.isfinite(state_dot.w_B_dot))
        assert np.all(np.isfinite(state_dot.delta_dot))
    
    def test_dynamics_finite_low_velocity(self, params):
        """Test dynamics handles low velocity (epsilon protection)."""
        state = State(
            p_I=np.array([0.0, 0.0, -500.0]),
            v_I=np.array([0.1, 0.0, 0.1]),  # Very low velocity
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.0, 0.0, 0.0]),
            delta=np.array([0.0, 0.0]),
            t=0.0
        )
        cmd = ControlCmd(delta_cmd=np.array([0.0, 0.0]))
        
        state_dot = dynamics(state, cmd, params)
        
        assert np.all(np.isfinite(state_dot.p_I_dot))
        assert np.all(np.isfinite(state_dot.v_I_dot))
        assert np.all(np.isfinite(state_dot.q_IB_dot))
        assert np.all(np.isfinite(state_dot.w_B_dot))
    
    def test_dynamics_finite_zero_velocity(self, params):
        """Test dynamics handles zero velocity."""
        state = State(
            p_I=np.array([0.0, 0.0, -500.0]),
            v_I=np.array([0.0, 0.0, 0.0]),  # Zero velocity
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.0, 0.0, 0.0]),
            delta=np.array([0.0, 0.0]),
            t=0.0
        )
        cmd = ControlCmd(delta_cmd=np.array([0.0, 0.0]))
        
        state_dot = dynamics(state, cmd, params)
        
        assert np.all(np.isfinite(state_dot.p_I_dot))
        assert np.all(np.isfinite(state_dot.v_I_dot))
    
    def test_dynamics_finite_random_states(self, params):
        """Test dynamics with random reasonable states."""
        rng = np.random.default_rng(42)
        
        for _ in range(100):
            # Random position
            p_I = rng.uniform(-1000, 1000, 3)
            p_I[2] = rng.uniform(-1000, -10)  # Ensure above ground
            
            # Random velocity
            v_I = rng.uniform(-30, 30, 3)
            
            # Random quaternion (normalized)
            q_IB = rng.standard_normal(4)
            q_IB = q_IB / np.linalg.norm(q_IB)
            
            # Random angular velocity
            w_B = rng.uniform(-1, 1, 3)
            
            # Random actuator state
            delta = rng.uniform(0, 1, 2)
            
            state = State(p_I=p_I, v_I=v_I, q_IB=q_IB, w_B=w_B, delta=delta, t=0.0)
            cmd = ControlCmd(delta_cmd=rng.uniform(0, 1, 2))
            
            state_dot = dynamics(state, cmd, params)
            
            assert np.all(np.isfinite(state_dot.p_I_dot)), f"p_I_dot not finite"
            assert np.all(np.isfinite(state_dot.v_I_dot)), f"v_I_dot not finite"
            assert np.all(np.isfinite(state_dot.q_IB_dot)), f"q_IB_dot not finite"
            assert np.all(np.isfinite(state_dot.w_B_dot)), f"w_B_dot not finite"


class TestAerodynamicAngles:
    """Test aerodynamic angle calculations."""
    
    def test_compute_aero_angles_forward_flight(self):
        """Test aero angles for straight forward flight."""
        v_B = np.array([10.0, 0.0, 0.0])  # Pure forward
        V, alpha, beta = compute_aero_angles(v_B)
        
        assert_allclose(V, 10.0, atol=1e-10)
        assert_allclose(alpha, 0.0, atol=1e-10)
        assert_allclose(beta, 0.0, atol=1e-10)
    
    def test_compute_aero_angles_with_aoa(self):
        """Test aero angles with angle of attack."""
        # 45 degree angle of attack
        v_B = np.array([10.0, 0.0, 10.0])
        V, alpha, beta = compute_aero_angles(v_B)
        
        assert_allclose(V, np.sqrt(200), atol=1e-10)
        assert_allclose(alpha, np.pi/4, atol=1e-10)
        assert_allclose(beta, 0.0, atol=1e-10)
    
    def test_compute_aero_angles_with_sideslip(self):
        """Test aero angles with sideslip."""
        v_B = np.array([10.0, 5.0, 0.0])  # Some sideslip
        V, alpha, beta = compute_aero_angles(v_B)
        
        expected_V = np.sqrt(125)
        expected_beta = np.arcsin(5.0 / expected_V)
        
        assert_allclose(V, expected_V, atol=1e-10)
        assert_allclose(beta, expected_beta, atol=1e-10)


class TestWindModel:
    """Test wind model functionality."""
    
    def test_wind_disabled(self):
        """Test that disabled wind returns zeros."""
        config = WindConfig(
            enable_steady=False,
            enable_gust=False,
            enable_colored=False
        )
        wind = WindModel(config)
        wind.reset()
        
        for t in np.linspace(0, 10, 100):
            w = wind.get_wind(t)
            assert_allclose(w, np.zeros(3), atol=1e-10)
    
    def test_steady_wind(self):
        """Test steady wind returns constant value."""
        steady = np.array([5.0, 2.0, -1.0])
        config = WindConfig(
            enable_steady=True,
            enable_gust=False,
            enable_colored=False,
            steady_wind=steady
        )
        wind = WindModel(config)
        wind.reset()
        
        for t in np.linspace(0, 10, 100):
            w = wind.get_wind(t)
            assert_allclose(w, steady, atol=1e-10)
    
    def test_wind_reproducibility(self):
        """Test that same seed gives same wind sequence."""
        config = WindConfig(
            enable_steady=True,
            enable_gust=True,
            enable_colored=True,
            seed=12345
        )
        
        wind1 = WindModel(config)
        wind1.reset()
        sequence1 = [wind1.get_wind(t, 0.1) for t in np.arange(0, 10, 0.1)]
        
        wind2 = WindModel(config)
        wind2.reset()
        sequence2 = [wind2.get_wind(t, 0.1) for t in np.arange(0, 10, 0.1)]
        
        for w1, w2 in zip(sequence1, sequence2):
            assert_allclose(w1, w2, atol=1e-10)


class TestSensorModel:
    """Test sensor model functionality."""
    
    def test_sensor_no_noise(self):
        """Test sensor with zero noise returns true values."""
        config = SensorConfig(
            position_noise_std=np.zeros(3),
            accel_noise_std=np.zeros(3),
            accel_bias=np.zeros(3),
            gyro_noise_std=np.zeros(3),
            gyro_bias=np.zeros(3)
        )
        sensor = SensorModel(config)
        params = Params()
        
        state = State(
            p_I=np.array([100.0, 200.0, -300.0]),
            v_I=np.array([10.0, 0.0, 2.0]),
            q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
            w_B=np.array([0.1, 0.2, 0.3]),
            delta=np.array([0.0, 0.0]),
            t=5.0
        )
        
        body_acc = np.array([1.0, 2.0, 3.0])
        meas = sensor.get_measurement(state, params, body_acc)
        
        assert_allclose(meas.position, state.p_I, atol=1e-10)
        assert_allclose(meas.body_acc, body_acc, atol=1e-10)
        assert_allclose(meas.body_ang_vel, state.w_B, atol=1e-10)
    
    def test_sensor_with_bias(self):
        """Test sensor bias is added correctly."""
        accel_bias = np.array([0.1, 0.2, 0.3])
        gyro_bias = np.array([0.01, 0.02, 0.03])
        
        config = SensorConfig(
            position_noise_std=np.zeros(3),
            accel_noise_std=np.zeros(3),
            accel_bias=accel_bias,
            gyro_noise_std=np.zeros(3),
            gyro_bias=gyro_bias
        )
        sensor = SensorModel(config)
        params = Params()
        
        state = State(
            p_I=np.array([0.0, 0.0, -100.0]),
            w_B=np.array([0.0, 0.0, 0.0]),
        )
        
        body_acc = np.array([0.0, 0.0, 0.0])
        meas = sensor.get_measurement(state, params, body_acc)
        
        assert_allclose(meas.body_acc, accel_bias, atol=1e-10)
        assert_allclose(meas.body_ang_vel, gyro_bias, atol=1e-10)


class TestStateOperations:
    """Test State class operations."""
    
    def test_state_copy(self):
        """Test state deep copy."""
        state = State(
            p_I=np.array([1.0, 2.0, 3.0]),
            v_I=np.array([4.0, 5.0, 6.0]),
        )
        
        state_copy = state.copy()
        state_copy.p_I[0] = 999.0
        
        # Original should be unchanged
        assert state.p_I[0] == 1.0
    
    def test_state_to_from_array(self):
        """Test state serialization roundtrip."""
        state = State(
            p_I=np.array([1.0, 2.0, 3.0]),
            v_I=np.array([4.0, 5.0, 6.0]),
            q_IB=np.array([0.5, 0.5, 0.5, 0.5]),
            w_B=np.array([0.1, 0.2, 0.3]),
            delta=np.array([0.4, 0.6]),
            t=10.0
        )
        
        arr = state.to_array()
        state_back = State.from_array(arr)
        
        assert_allclose(state.p_I, state_back.p_I)
        assert_allclose(state.v_I, state_back.v_I)
        assert_allclose(state.q_IB, state_back.q_IB)
        assert_allclose(state.w_B, state_back.w_B)
        assert_allclose(state.delta, state_back.delta)
        assert state.t == state_back.t
    
    def test_state_is_on_ground(self):
        """Test ground detection."""
        state_air = State(p_I=np.array([0.0, 0.0, -100.0]))
        state_ground = State(p_I=np.array([0.0, 0.0, 1.0]))
        
        assert not state_air.is_on_ground
        assert state_ground.is_on_ground
    
    def test_state_altitude(self):
        """Test altitude calculation."""
        state = State(p_I=np.array([0.0, 0.0, -500.0]))
        assert state.altitude == 500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
