#!/usr/bin/env python3
"""
Twistor Theory Visualization Module
====================================
Generates plots and visualizations of twistor geometry and quantum dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Import from main simulation
from twistor_simulation import (
    Twistor, TwistorSpace, TwistorQuantumState, 
    TwistorHamiltonian, MHVAmplitudes, SpinorAlgebra,
    PenroseTransform
)


def plot_helicity_spectrum(twistors, title="Twistor Helicity Spectrum"):
    """Plot the helicity distribution of a set of twistors."""
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    helicities = [Z.helicity for Z in twistors]
    
    # Create stem plot
    markerline, stemlines, baseline = ax.stem(
        range(len(helicities)), helicities,
        linefmt='b-', markerfmt='bo', basefmt='k-'
    )
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhspan(-0.5, 0.5, alpha=0.1, color='green', label='Near-null region')
    
    ax.set_xlabel('Twistor Index', fontsize=12)
    ax.set_ylabel('Helicity s', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add helicity labels
    for i, h in enumerate(helicities):
        ax.annotate(f's={h:.2f}', (i, h), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_spinor_brackets(spinors, title="Spinor Bracket Matrix |⟨ij⟩|"):
    """Visualize the matrix of spinor brackets."""
    
    n = len(spinors)
    brackets = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            brackets[i, j] = MHVAmplitudes.spinor_bracket(spinors[i], spinors[j])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Magnitude
    im1 = ax1.imshow(np.abs(brackets), cmap='viridis')
    ax1.set_title('|⟨ij⟩| Magnitude', fontsize=12)
    ax1.set_xlabel('j')
    ax1.set_ylabel('i')
    plt.colorbar(im1, ax=ax1)
    
    # Phase
    im2 = ax2.imshow(np.angle(brackets), cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax2.set_title('arg(⟨ij⟩) Phase', fontsize=12)
    ax2.set_xlabel('j')
    ax2.set_ylabel('i')
    plt.colorbar(im2, ax=ax2, label='Radians')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_quantum_evolution(psi_0, hamiltonian, times, title="Quantum Evolution"):
    """Plot the time evolution of quantum state probabilities."""
    
    n_states = len(psi_0.coefficients)
    probabilities = []
    helicities = []
    
    for t in times:
        psi_t = psi_0.evolve(hamiltonian, t)
        probs = np.abs(psi_t.coefficients) ** 2
        probabilities.append(probs)
        
        # Compute helicity expectation
        helicity_op = TwistorHamiltonian.helicity_operator(n_states)
        h_exp = psi_t.expectation_value(helicity_op).real
        helicities.append(h_exp)
    
    probabilities = np.array(probabilities)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Probability evolution
    for i in range(n_states):
        ax1.plot(times, probabilities[:, i], label=f'|c_{i}|²', linewidth=2)
    
    ax1.set_ylabel('Probability |cᵢ|²', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Helicity expectation
    ax2.plot(times, helicities, 'k-', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time t', fontsize=12)
    ax2.set_ylabel('⟨s⟩ Helicity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_null_geodesics_2d(twistors, title="Null Geodesics Projection"):
    """Plot 2D projection of null geodesics in spacetime."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(twistors)))
    
    for i, Z in enumerate(twistors):
        try:
            point, direction = Z.to_spacetime_line()
            
            # Draw the null line (projection onto t-x plane)
            t_vals = np.linspace(-2, 2, 100)
            x_vals = point[0] + t_vals * direction[0]
            y_vals = point[1] + t_vals * direction[1]
            
            ax.plot(x_vals, y_vals, color=colors[i], 
                   label=f'Z_{i+1} (s={Z.helicity:.2f})', linewidth=2)
            ax.scatter([point[0]], [point[1]], color=colors[i], s=50, zorder=5)
            
        except Exception:
            continue
    
    ax.set_xlabel('x⁰ (time)', fontsize=12)
    ax.set_ylabel('x¹ (space)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Draw light cone from origin
    t = np.linspace(-2, 2, 100)
    ax.plot(t, t, 'k--', alpha=0.3, label='Light cone')
    ax.plot(t, -t, 'k--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_mhv_amplitude_scan(spinors, title="MHV Amplitude Variation"):
    """Plot how MHV amplitude changes with spinor phase rotation."""
    
    phases = np.linspace(0, 2*np.pi, 100)
    amplitudes = []
    
    for phi in phases:
        # Rotate first spinor by phase phi
        rotated_spinors = spinors.copy()
        rotated_spinors[0] = spinors[0] * np.exp(1j * phi)
        
        amp = MHVAmplitudes.parke_taylor_amplitude(rotated_spinors, (0, 1))
        amplitudes.append(np.abs(amp))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(phases, amplitudes, 'b-', linewidth=2)
    ax.set_xlabel('Phase rotation φ', fontsize=12)
    ax.set_ylabel('|A₄|', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_twistor_sphere(n_points=50, title="Twistor Space on S²"):
    """
    Visualize twistor space using the Hopf fibration projection to S².
    
    The projective twistor space CP³ can be visualized via its relationship
    to the celestial sphere of an observer.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate points on the celestial sphere
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2*np.pi, n_points)
    theta, phi = np.meshgrid(theta, phi)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # Color by helicity (simulated)
    colors = np.cos(theta)  # Helicity varies with polar angle
    
    ax.plot_surface(x, y, z, facecolors=plt.cm.coolwarm(0.5 + 0.5*colors),
                   alpha=0.7, rstride=1, cstride=1, linewidth=0)
    
    # Add some twistor points
    np.random.seed(42)
    for i in range(10):
        Z = TwistorSpace.create_random_twistor()
        # Project to S² using the spinor direction
        pi_norm = Z.pi / np.linalg.norm(Z.pi)
        
        # Stereographic coordinates
        zeta = pi_norm[0] / (1 + 1e-10 - pi_norm[1]) if np.abs(1 - pi_norm[1]) > 0.01 else 1e10
        
        # Map to S²
        r2 = np.abs(zeta)**2
        xs = 2 * zeta.real / (1 + r2)
        ys = 2 * zeta.imag / (1 + r2)
        zs = (r2 - 1) / (1 + r2)
        
        color = 'red' if Z.helicity > 0 else 'blue'
        ax.scatter([xs], [ys], [zs], c=color, s=100, alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    
    # Add color bar legend
    ax.text2D(0.05, 0.95, "Red: s > 0 (positive helicity)", transform=ax.transAxes, color='red')
    ax.text2D(0.05, 0.90, "Blue: s < 0 (negative helicity)", transform=ax.transAxes, color='blue')
    
    return fig


def plot_field_contour(twistor_func, x_range=(-2, 2), y_range=(-2, 2), 
                       n_points=50, title="Massless Field |φ(x)|"):
    """Plot the magnitude of a massless field in a spacetime slice."""
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    field_vals = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            spacetime_point = np.array([X[i,j], Y[i,j], 0, 0])
            phi = PenroseTransform.contour_integral_scalar(
                twistor_func, spacetime_point, n_points=50
            )
            field_vals[i, j] = np.abs(phi)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.contourf(X, Y, field_vals, levels=20, cmap='plasma')
    plt.colorbar(im, ax=ax, label='|φ|')
    
    ax.set_xlabel('x⁰', fontsize=12)
    ax.set_ylabel('x¹', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def create_comprehensive_visualization():
    """Generate a comprehensive set of visualizations."""
    
    np.random.seed(42)
    
    print("Generating twistor theory visualizations...")
    print("=" * 50)
    
    # Create twistors
    twistors = [TwistorSpace.create_random_twistor() for _ in range(8)]
    
    # Create spinors for amplitude calculations
    spinors = [
        np.array([1.0, 0.0], dtype=complex),
        np.array([0.0, 1.0], dtype=complex),
        np.array([1.0, 1.0], dtype=complex) / np.sqrt(2),
        np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
    ]
    
    # 1. Helicity spectrum
    print("1. Creating helicity spectrum plot...")
    fig1 = plot_helicity_spectrum(twistors)
    fig1.savefig('helicity_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Spinor brackets
    print("2. Creating spinor bracket matrix...")
    fig2 = plot_spinor_brackets(spinors)
    fig2.savefig('spinor_brackets.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Quantum evolution
    print("3. Creating quantum evolution plot...")
    n_states = 5
    basis = [TwistorSpace.create_random_twistor() for _ in range(n_states)]
    initial = np.array([1, 1, 0, 0, 0], dtype=complex) / np.sqrt(2)
    psi_0 = TwistorQuantumState(initial, basis).normalize()
    
    H = TwistorHamiltonian.free_particle_hamiltonian(basis)
    H += TwistorHamiltonian.interaction_hamiltonian(basis, 0.1)
    
    times = np.linspace(0, 10, 100)
    fig3 = plot_quantum_evolution(psi_0, H, times)
    fig3.savefig('quantum_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. MHV amplitude scan
    print("4. Creating MHV amplitude variation plot...")
    fig4 = plot_mhv_amplitude_scan(spinors)
    fig4.savefig('mhv_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Twistor sphere
    print("5. Creating twistor sphere visualization...")
    fig5 = plot_twistor_sphere()
    fig5.savefig('twistor_sphere.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Combined summary figure
    print("6. Creating summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Subplot 1: Helicities
    helicities = [Z.helicity for Z in twistors]
    axes[0, 0].bar(range(len(helicities)), helicities, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Twistor Index')
    axes[0, 0].set_ylabel('Helicity s')
    axes[0, 0].set_title('Helicity Distribution')
    
    # Subplot 2: Spinor bracket magnitudes
    n = len(spinors)
    brackets = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            brackets[i, j] = np.abs(MHVAmplitudes.spinor_bracket(spinors[i], spinors[j]))
    im = axes[0, 1].imshow(brackets, cmap='viridis')
    axes[0, 1].set_title('|⟨ij⟩| Bracket Matrix')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Subplot 3: Probability evolution
    probs = []
    for t in times:
        psi_t = psi_0.evolve(H, t)
        probs.append(np.abs(psi_t.coefficients) ** 2)
    probs = np.array(probs)
    
    for i in range(n_states):
        axes[1, 0].plot(times, probs[:, i], label=f'|c_{i}|²')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Quantum State Evolution')
    axes[1, 0].legend()
    
    # Subplot 4: MHV amplitude
    phases = np.linspace(0, 2*np.pi, 50)
    amps = []
    for phi in phases:
        rot_spinors = spinors.copy()
        rot_spinors[0] = spinors[0] * np.exp(1j * phi)
        amp = MHVAmplitudes.parke_taylor_amplitude(rot_spinors, (0, 1))
        amps.append(np.abs(amp))
    
    axes[1, 1].plot(phases, amps, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Phase φ')
    axes[1, 1].set_ylabel('|A₄|')
    axes[1, 1].set_title('MHV Amplitude vs Phase')
    
    plt.suptitle('Quantum Twistor Theory: Key Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig('twistor_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("=" * 50)
    print("Visualizations saved:")
    print("  - helicity_spectrum.png")
    print("  - spinor_brackets.png")
    print("  - quantum_evolution.png")
    print("  - mhv_amplitude.png")
    print("  - twistor_sphere.png")
    print("  - twistor_summary.png")
    print("=" * 50)


if __name__ == "__main__":
    create_comprehensive_visualization()
