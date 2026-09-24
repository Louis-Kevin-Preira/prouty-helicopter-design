"""Chapter 5 -- promoted names and defaults consistent across the closed-form groups.

The trim-based groups (G2b, G2e, G3, G4, G5 optimum, G6) carry rotor states of
their own (alpha_TPP, CT_sigma, ...) and are assembled with explicit paths.
"""
import openmdao.api as om

from prouty.special_performance import (TurnsPullupsGroup, RotorSpeedDecayGroup,
                                        AutorotativeIndicesGroup, ZoomGlideGroup,
                                        HeightVelocityGroup, TakeoffDistanceGroup, TowingGroup)


def test_promoted_defaults_are_consistent_across_the_chapter():
    p = om.Problem()
    m = p.model
    m.add_subsystem('g1', TurnsPullupsGroup(), promotes=['*'])
    m.add_subsystem('g2a', RotorSpeedDecayGroup(), promotes=['*'])
    m.add_subsystem('g2f', AutorotativeIndicesGroup(), promotes=['*'])
    m.add_subsystem('g2c', ZoomGlideGroup(), promotes=['*'])
    m.add_subsystem('g2d', HeightVelocityGroup(num_points=3), promotes=['*'])
    m.add_subsystem('g5', TakeoffDistanceGroup(), promotes=['*'])
    m.add_subsystem('g7', TowingGroup(), promotes=['*'])
    p.setup()
    p.final_setup()
